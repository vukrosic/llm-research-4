import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Dict
import warnings
import os
import pickle
import wandb

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    print("🚀 Triton available - will use for performance optimization")
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️  Triton not available - using PyTorch implementations only")

warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 48  # Increased from 24 to 48
    max_steps: int = 100

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 500  # Reduced from 2000 for faster training
    max_tokens: int = 200000  # Reduced from 500000 for faster training

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    # Weights & Biases
    wandb_project: str = "llm-med"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_every: int = 10

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Performance optimization settings
        self.use_triton_rmsnorm: bool = True
        self.use_triton_rotary: bool = True
        self.benchmark_kernels: bool = True  # Auto-benchmark and select best

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

# ============= TRITON KERNELS =============

# Define Triton kernels and classes conditionally
if TRITON_AVAILABLE:
    @triton.jit
    def rms_norm_fwd_kernel(
        X,  # input
        Y,  # output
        W,  # weight (gamma)
        rstd,  # reciprocal standard deviation
        stride,
        N,  # number of columns
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm forward kernel - optimized version"""
        row = tl.program_id(0)
        X += row * stride
        Y += row * stride
        
        # Compute variance in a single pass with better memory access
        var = 0.0
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            var += tl.sum(x * x, axis=0)
        
        var = var / N
        rstd_val = 1 / tl.sqrt(var + eps)
        tl.store(rstd + row, rstd_val)
        
        # Normalize and apply weight in a single pass
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * rstd_val * w
            tl.store(Y + cols, y, mask=mask)

    @triton.jit
    def rms_norm_bwd_kernel(
        X, W, DY, DX, DW,
        rstd,
        stride,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """RMSNorm backward kernel"""
        row = tl.program_id(0)
        X += row * stride
        DY += row * stride
        DX += row * stride
        
        rstd_val = tl.load(rstd + row)
        
        # Compute gradients
        _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            
            # Accumulate for dW
            tl.atomic_add(DW + cols, (x * rstd_val * dy).to(tl.float32), mask=mask)
            
            # Compute sum for dx
            _sum += x * dy * w
        
        sum_val = tl.sum(_sum, axis=0) / N
        
        # Compute dx
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            
            dx = rstd_val * w * (dy - x * sum_val * rstd_val * rstd_val)
            tl.store(DX + cols, dx, mask=mask)

    class TritonRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps=1e-5):
            assert x.is_cuda and weight.is_cuda
            M, N = x.shape
            y = torch.empty_like(x)
            rstd = torch.empty(M, dtype=torch.float32, device=x.device)
            
            # Use larger block size for better GPU utilization
            BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
            grid = (M,)
            
            rms_norm_fwd_kernel[grid](
                x, y, weight, rstd,
                x.stride(0),
                N,
                eps=eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            ctx.save_for_backward(x, weight, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.N = N
            return y
        
        @staticmethod
        def backward(ctx, dy):
            x, weight, rstd = ctx.saved_tensors
            M, N = x.shape
            
            dx = torch.empty_like(x)
            dw = torch.zeros_like(weight)
            
            grid = (M,)
            
            rms_norm_bwd_kernel[grid](
                x, weight, dy, dx, dw,
                rstd,
                x.stride(0),
                N,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
            )
            
            return dx, dw, None

    @triton.jit
    def rotary_kernel(
        Q, K,
        cos, sin,
        seqlen,
        d_head,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Apply rotary position embeddings"""
        batch_head_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        
        # Calculate offsets
        q_offset = batch_head_idx * seqlen * d_head + seq_idx * d_head
        k_offset = batch_head_idx * seqlen * d_head + seq_idx * d_head
        
        # Process in chunks
        for i in range(0, d_head // 2, BLOCK_SIZE):
            idx = i + tl.arange(0, BLOCK_SIZE)
            mask = idx < d_head // 2
            
            # Load Q and K values
            q1 = tl.load(Q + q_offset + idx, mask=mask, other=0.0)
            q2 = tl.load(Q + q_offset + idx + d_head // 2, mask=mask, other=0.0)
            k1 = tl.load(K + k_offset + idx, mask=mask, other=0.0)
            k2 = tl.load(K + k_offset + idx + d_head // 2, mask=mask, other=0.0)
            
            # Load cos and sin
            c = tl.load(cos + seq_idx * (d_head // 2) + idx, mask=mask, other=0.0)
            s = tl.load(sin + seq_idx * (d_head // 2) + idx, mask=mask, other=0.0)
            
            # Apply rotation
            q1_new = q1 * c + q2 * s
            q2_new = q1 * (-s) + q2 * c
            k1_new = k1 * c + k2 * s
            k2_new = k1 * (-s) + k2 * c
            
            # Store results
            tl.store(Q + q_offset + idx, q1_new, mask=mask)
            tl.store(Q + q_offset + idx + d_head // 2, q2_new, mask=mask)
            tl.store(K + k_offset + idx, k1_new, mask=mask)
            tl.store(K + k_offset + idx + d_head // 2, k2_new, mask=mask)

    class TritonRotary(nn.Module):
        def __init__(self, dim: int, max_seq_len: int):
            super().__init__()
            self.dim = dim
            
            # Precompute cos and sin
            angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
            t = torch.arange(max_seq_len, dtype=torch.float32)
            theta = torch.einsum("i,j -> ij", t, angular_freq)
            self.register_buffer('cos', theta.cos(), persistent=False)
            self.register_buffer('sin', theta.sin(), persistent=False)
        
        def forward(self, q, k):
            batch_size, n_heads, seq_len, d_head = q.shape
            
            # Reshape for kernel
            q = q.reshape(batch_size * n_heads, seq_len, d_head)
            k = k.reshape(batch_size * n_heads, seq_len, d_head)
            
            # Apply rotary embeddings
            grid = (batch_size * n_heads, seq_len)
            BLOCK_SIZE = triton.next_power_of_2(d_head // 2)
            
            rotary_kernel[grid](
                q, k,
                self.cos, self.sin,
                seq_len, d_head,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            # Reshape back
            q = q.reshape(batch_size, n_heads, seq_len, d_head)
            k = k.reshape(batch_size, n_heads, seq_len, d_head)
            
            return q, k

# ============= PYTORCH INTEGRATION =============

# Define wrapper classes that work whether Triton is available or not
class TritonRMSNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        if TRITON_AVAILABLE:
            orig_shape = x.shape
            x = x.view(-1, orig_shape[-1])
            
            # Ensure weight dtype matches input
            if self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            else:
                weight = self.weight
                
            x = TritonRMSNorm.apply(x, weight, self.eps)
            return x.view(orig_shape)
        else:
            # Fallback to PyTorch RMSNorm
            return F.layer_norm(x, (x.shape[-1],), self.weight, None, self.eps)

class TritonRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        
        # Precompute cos and sin
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
    
    def forward(self, q, k):
        if TRITON_AVAILABLE:
            batch_size, n_heads, seq_len, d_head = q.shape
            
            # Reshape for kernel
            q = q.reshape(batch_size * n_heads, seq_len, d_head)
            k = k.reshape(batch_size * n_heads, seq_len, d_head)
            
            # Apply rotary embeddings
            grid = (batch_size * n_heads, seq_len)
            BLOCK_SIZE = triton.next_power_of_2(d_head // 2)
            
            rotary_kernel[grid](
                q, k,
                self.cos, self.sin,
                seq_len, d_head,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            # Reshape back
            q = q.reshape(batch_size, n_heads, seq_len, d_head)
            k = k.reshape(batch_size, n_heads, seq_len, d_head)
            
            return q, k
        else:
            # Fallback to PyTorch implementation
            batch_size, n_heads, seq_len, d_head = q.shape
            cos = self.cos[:seq_len, :d_head//2].unsqueeze(0).unsqueeze(2)
            sin = self.sin[:seq_len, :d_head//2].unsqueeze(0).unsqueeze(2)
            
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            
            q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
            
            return q_rot, k_rot

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/thebluescrubs_tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("openmed-community/TheBlueScrubs-v1-fixed", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        # TheBlueScrubs dataset has 'text' field
        text = item.get("text", "")
        if text:
            texts.append(text[:3000])
            # Print first few documents for inspection
            if i < 3:
                print(f"\n📄 Document {i+1} (first 200 chars):")
                print(f"   {text[:200]}...")
                print(f"   Length: {len(text)} characters")

    print(f"Loaded {len(texts)} documents from TheBlueScrubs dataset")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    total_chars = 0
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        total_chars += len(text)

    tokens = all_tokens[:config.max_tokens]
    print(f"📊 Dataset Statistics:")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Total tokens: {len(all_tokens):,}")
    print(f"   Using tokens: {len(tokens):,}")
    print(f"   Average tokens per document: {len(all_tokens) // len(texts):,}")
    print(f"   Vocabulary size: {tokenizer.vocab_size:,}")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    
    # Show sample tokenization
    if texts:
        sample_text = texts[0][:100]
        sample_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
        sample_decoded = tokenizer.decode(sample_tokens[:20])
        print(f"\n🔍 Sample Tokenization:")
        print(f"   Original: '{sample_text}...'")
        print(f"   Tokens: {sample_tokens[:20]}")
        print(f"   Decoded: '{sample_decoded}...'")
    
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, use_triton_rotary: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Choose Rotary implementation based on performance
        if use_triton_rotary and TRITON_AVAILABLE:
            self.rotary = TritonRotary(self.d_k, max_seq_len)
        else:
            self.rotary = Rotary(self.d_k, max_seq_len)
            
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1, use_triton_rmsnorm: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, use_triton_rotary=use_triton_rmsnorm)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Choose RMSNorm implementation based on performance
        if use_triton_rmsnorm and TRITON_AVAILABLE:
            self.norm1 = TritonRMSNormLayer(d_model)
            self.norm2 = TritonRMSNormLayer(d_model)
        else:
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout,
                use_triton_rmsnorm=config.use_triton_rmsnorm
            )
            for _ in range(config.n_layers)
        ])

        # Choose final norm implementation based on performance
        if config.use_triton_rmsnorm and TRITON_AVAILABLE:
            self.norm = TritonRMSNormLayer(config.d_model)
        else:
            self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

# ============= KERNEL BENCHMARKING AND AUTO-SELECTION =============

def benchmark_rms_norm(batch_size: int, seq_len: int, d_model: int, num_runs: int = 50) -> Dict[str, float]:
    """Benchmark RMSNorm implementations"""
    device = torch.device('cuda')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    
    # PyTorch RMSNorm
    pytorch_norm = nn.RMSNorm(d_model).to(device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = pytorch_norm(x)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_norm(x)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    if TRITON_AVAILABLE:
        # Triton RMSNorm
        triton_norm = TritonRMSNormLayer(d_model).to(device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = triton_norm(x)
        
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start_time = time.time()
        for _ in range(num_runs):
            _ = triton_norm(x)
        torch.cuda.synchronize()
        triton_time = time.time() - start_time
        
        return {
            'pytorch_time': pytorch_time / num_runs * 1000,  # ms
            'triton_time': triton_time / num_runs * 1000,    # ms
            'speedup': pytorch_time / triton_time,
            'use_triton': triton_time < pytorch_time
        }
    else:
        return {
            'pytorch_time': pytorch_time / num_runs * 1000,
            'triton_time': float('inf'),
            'speedup': 1.0,
            'use_triton': False
        }

def benchmark_rotary(batch_size: int, seq_len: int, n_heads: int, d_head: int, num_runs: int = 50) -> Dict[str, float]:
    """Benchmark Rotary implementations"""
    device = torch.device('cuda')
    
    # Create test data
    q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=torch.float16)
    
    # PyTorch Rotary
    pytorch_rotary = Rotary(d_head, seq_len).to(device)
    
    # Warmup
    for _ in range(10):
        _ = pytorch_rotary(q)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_rotary(q)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    if TRITON_AVAILABLE:
        # Triton Rotary
        triton_rotary = TritonRotary(d_head, seq_len).to(device)
        
        # Warmup
        for _ in range(10):
            _ = triton_rotary(q, k)
        
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start_time = time.time()
        for _ in range(num_runs):
            _ = triton_rotary(q, k)
        torch.cuda.synchronize()
        triton_time = time.time() - start_time
        
        return {
            'pytorch_time': pytorch_time / num_runs * 1000,
            'triton_time': triton_time / num_runs * 1000,
            'speedup': pytorch_time / triton_time,
            'use_triton': triton_time < pytorch_time
        }
    else:
        return {
            'pytorch_time': pytorch_time / num_runs * 1000,
            'triton_time': float('inf'),
            'speedup': 1.0,
            'use_triton': False
        }

def auto_select_kernels(config: ModelConfig) -> Dict[str, bool]:
    """Automatically benchmark and select the best kernels for the current configuration"""
    if not config.benchmark_kernels or not TRITON_AVAILABLE:
        return {
            'use_triton_rmsnorm': False,
            'use_triton_rotary': False
        }
    
    print("🔍 Auto-benchmarking kernels for optimal performance...")
    
    # Test with current configuration
    batch_size = min(config.batch_size, 16)  # Use smaller batch for benchmarking
    seq_len = min(config.max_seq_len, 256)
    d_model = config.d_model
    n_heads = config.n_heads
    d_head = d_model // n_heads
    
    print(f"  Testing with: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    
    # Benchmark RMSNorm
    print("  Benchmarking RMSNorm...")
    rms_results = benchmark_rms_norm(batch_size, seq_len, d_model)
    print(f"    PyTorch: {rms_results['pytorch_time']:.3f} ms")
    if TRITON_AVAILABLE:
        print(f"    Triton:  {rms_results['triton_time']:.3f} ms")
        print(f"    Speedup: {rms_results['speedup']:.2f}x")
        print(f"    Using:   {'Triton' if rms_results['use_triton'] else 'PyTorch'}")
    
    # Benchmark Rotary
    print("  Benchmarking Rotary...")
    rotary_results = benchmark_rotary(batch_size, seq_len, n_heads, d_head)
    print(f"    PyTorch: {rotary_results['pytorch_time']:.3f} ms")
    if TRITON_AVAILABLE:
        print(f"    Triton:  {rotary_results['triton_time']:.3f} ms")
        print(f"    Speedup: {rotary_results['speedup']:.2f}x")
        print(f"    Using:   {'Triton' if rotary_results['use_triton'] else 'PyTorch'}")
    
    # Update config with best choices
    config.use_triton_rmsnorm = rms_results['use_triton']
    config.use_triton_rotary = rotary_results['use_triton']
    
    return {
        'use_triton_rmsnorm': rms_results['use_triton'],
        'use_triton_rotary': rotary_results['use_triton']
    }

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with Muon optimizer"""
    print(f"\n🚀 Training Small model with Muon optimizer")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config={
            "architecture": f"{config.d_model}d-{config.n_layers}L-{config.n_heads}H-{config.d_ff}ff",
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "max_seq_len": config.max_seq_len,
            "num_documents": config.num_documents,
            "max_tokens": config.max_tokens,
            "muon_lr": config.muon_lr,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "grad_clip": config.grad_clip,
            "use_amp": config.use_amp,
            "eval_every": config.eval_every,
            "eval_steps": config.eval_steps,
        }
    )

    # Initialize model
    set_seed(42)
    
    # Auto-select best kernels based on performance
    if config.benchmark_kernels:
        kernel_selection = auto_select_kernels(config)
        print(f"  🚀 Kernel selection: RMSNorm={'Triton' if kernel_selection['use_triton_rmsnorm'] else 'PyTorch'}, Rotary={'Triton' if kernel_selection['use_triton_rotary'] else 'PyTorch'}")
    
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")
    
    # Log model architecture to wandb
    wandb.run.summary["total_parameters"] = total_params
    wandb.run.summary["model_architecture"] = f"{config.d_model}d-{config.n_layers}L-{config.n_heads}H-{config.d_ff}ff"
    
    # Log kernel selection
    if config.benchmark_kernels:
        wandb.run.summary["use_triton_rmsnorm"] = config.use_triton_rmsnorm
        wandb.run.summary["use_triton_rotary"] = config.use_triton_rotary

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    # Track metrics for wandb
    running_loss = 0
    running_accuracy = 0
    running_perplexity = 0
    metric_steps = 0

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}'
                })

            # Track running metrics for wandb
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == y).float().mean().item()
                current_loss = loss.item() * config.gradient_accumulation_steps
                perplexity = math.exp(min(current_loss, 20))
                
                running_loss += current_loss
                running_accuracy += accuracy
                running_perplexity += perplexity
                metric_steps += 1

            # Log to wandb every few steps
            if step % config.log_every == 0 and step > 0:
                avg_loss = running_loss / metric_steps
                avg_accuracy = running_accuracy / metric_steps
                avg_perplexity = running_perplexity / metric_steps
                
                wandb.log({
                    "train/loss": avg_loss,
                    "train/accuracy": avg_accuracy,
                    "train/perplexity": avg_perplexity,
                    "train/learning_rate": optimizers[0].param_groups[0]["lr"],
                    "train/gradient_norm": grad_norm if 'grad_norm' in locals() else None,
                    "train/step": step,
                })
                
                # Reset running metrics
                running_loss = 0
                running_accuracy = 0
                running_perplexity = 0
                metric_steps = 0

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}")

                # Log validation metrics to wandb
                wandb.log({
                    "val/loss": eval_metrics['val_loss'],
                    "val/accuracy": eval_metrics['val_accuracy'],
                    "val/perplexity": eval_metrics['val_perplexity'],
                    "val/step": step,
                })

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']
                    # Save best model checkpoint locally (not to wandb)
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_states': [opt.state_dict() for opt in optimizers],
                        'scheduler_states': [sched.state_dict() for sched in schedulers],
                        'config': config,
                        'best_val_loss': best_val_loss,
                    }, f"best_model_step_{step}.pt")
                    
                    print(f"💾 Saved best model checkpoint to best_model_step_{step}.pt")

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")
    
    # Log final training time
    wandb.run.summary["training_time_seconds"] = training_time

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")
    
    # Log final metrics to wandb
    wandb.log({
        "final/val_loss": final_eval['val_loss'],
        "final/val_accuracy": final_eval['val_accuracy'],
        "final/val_perplexity": final_eval['val_perplexity'],
    })
    
    # Save final model locally (not to wandb)
    final_model_path = "final_model.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sched.state_dict() for sched in schedulers],
        'config': config,
        'final_val_loss': final_eval['val_loss'],
    }, final_model_path)
    
    print(f"💾 Saved final model to {final_model_path}")
    
    # Finish wandb run
    wandb.finish()

    return model, final_eval

def manual_kernel_benchmark():
    """Manual function to benchmark kernels independently"""
    if not TRITON_AVAILABLE:
        print("❌ Triton not available. Cannot run kernel benchmark.")
        return
    
    print("🔍 Manual Kernel Benchmark")
    print("=" * 40)
    
    # Test configurations
    configs = [
        {'batch_size': 16, 'seq_len': 256, 'd_model': 384, 'n_heads': 8},
        {'batch_size': 32, 'seq_len': 512, 'd_model': 768, 'n_heads': 12},
        {'batch_size': 64, 'seq_len': 1024, 'd_model': 1024, 'n_heads': 16},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n Configuration {i+1}: {config['batch_size']}x{config['seq_len']}x{config['d_model']}")
        print("-" * 50)
        
        # RMSNorm benchmark
        rms_results = benchmark_rms_norm(
            config['batch_size'], config['seq_len'], config['d_model']
        )
        print(f"  RMSNorm:")
        print(f"    PyTorch: {rms_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {rms_results['triton_time']:.3f} ms")
        print(f"    Speedup: {rms_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if rms_results['use_triton'] else 'PyTorch'}")
        
        # Rotary benchmark
        d_head = config['d_model'] // config['n_heads']
        rotary_results = benchmark_rotary(
            config['batch_size'], config['seq_len'], config['n_heads'], d_head
        )
        print(f"  Rotary:")
        print(f"    PyTorch: {rotary_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {rotary_results['triton_time']:.3f} ms")
        print(f"    Speedup: {rotary_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if rotary_results['use_triton'] else 'PyTorch'}")

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    
    # Set wandb run name if not specified
    if config.wandb_run_name is None:
        config.wandb_run_name = f"llm-med-{config.d_model}d-{config.n_layers}L-{config.n_heads}H"
    
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")
    print(f"   Dataset: TheBlueScrubs-v1-fixed (medical text)")
    print(f"   Weights & Biases: {config.wandb_project}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    print(f"🏆 Check your Weights & Biases dashboard for detailed training logs!")