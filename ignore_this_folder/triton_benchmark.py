import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============= TRITON KERNELS =============

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

@triton.jit
def fused_silu_mul_kernel(
    X, W1, W3, Y,
    M, N, K,
    stride_x, stride_w1, stride_w3, stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused SiLU gated linear unit: Y = (X @ W1) * silu(X @ W3)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = rm < M
    mask_n = rn < N
    
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        
        # Load X block
        x = tl.load(
            X + rm[:, None] * stride_x + rk[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load W1 and W3 blocks
        w1 = tl.load(
            W1 + rk[:, None] * stride_w1 + rn[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        w3 = tl.load(
            W3 + rk[:, None] * stride_w3 + rn[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )
        
        acc1 += tl.dot(x, w1)
        acc3 += tl.dot(x, w3)
    
    # Apply SiLU and gating
    silu_acc3 = acc3 * tl.sigmoid(acc3)
    result = acc1 * silu_acc3
    
    # Store result
    tl.store(
        Y + rm[:, None] * stride_y + rn[None, :],
        result,
        mask=mask_m[:, None] & mask_n[None, :],
    )

@triton.jit
def newton_schulz_kernel(
    G, X_out,
    M, N,
    steps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Newton-Schulz iteration for orthogonalization"""
    pid = tl.program_id(0)
    
    # Constants from the paper
    a = 3.4445
    b = -4.7750
    c = 2.0315
    
    # Initialize X as normalized G
    norm = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, M * N, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < M * N
        g_val = tl.load(G + idx, mask=mask, other=0.0).to(tl.float32)
        norm += tl.sum(g_val * g_val)
    
    norm = tl.sqrt(norm[0] + 1e-7)
    
    # Normalize and store initial X
    for i in range(0, M * N, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < M * N
        g_val = tl.load(G + idx, mask=mask, other=0.0).to(tl.float32)
        x_val = g_val / norm
        tl.store(X_out + idx, x_val, mask=mask)

# ============= PYTORCH INTEGRATION =============

class TritonRMSNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        
        # Ensure weight dtype matches input
        if self.weight.dtype != x.dtype:
            weight = self.weight.to(x.dtype)
        else:
            weight = self.weight
            
        x = TritonRMSNorm.apply(x, weight, self.eps)
        return x.view(orig_shape)

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

class TritonGatedMLP(nn.Module):
    """Fused SiLU-gated MLP using Triton"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        self.w3 = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
    
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        
        # Ensure weight dtypes match input
        w1 = self.w1.to(x.dtype) if self.w1.dtype != x.dtype else self.w1
        w2 = self.w2.to(x.dtype) if self.w2.dtype != x.dtype else self.w2
        w3 = self.w3.to(x.dtype) if self.w3.dtype != x.dtype else self.w3
        
        # Intermediate output
        intermediate = torch.empty(B * T, w1.shape[1], device=x.device, dtype=x.dtype)
        
        # Grid and block sizes
        M, K = x_flat.shape
        N = w1.shape[1]
        
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 32
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        # Fused gate computation
        fused_silu_mul_kernel[grid](
            x_flat, w1, w3, intermediate,
            M, N, K,
            x_flat.stride(0), w1.stride(0), w3.stride(0), intermediate.stride(0),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        
        # Final projection
        output = F.linear(intermediate, w2.t())
        return output.view(B, T, -1)

def zeropower_via_triton(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz using Triton kernel"""
    assert G.is_cuda
    orig_shape = G.shape
    G = G.reshape(-1)
    
    X = torch.empty_like(G)
    M, N = orig_shape[-2], orig_shape[-1] if len(orig_shape) > 1 else 1, G.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    newton_schulz_kernel[grid](
        G, X,
        M, N,
        steps=steps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return X.reshape(orig_shape)

# ============= ORIGINAL IMPLEMENTATIONS (FROM LLM.PY) =============

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

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))
    
    def to(self, device=None, dtype=None, non_blocking=False):
        # Ensure weights match input dtype
        if dtype is not None:
            self.linear1.weight.data = self.linear1.weight.data.to(dtype)
            self.linear2.weight.data = self.linear2.weight.data.to(dtype)
        return super().to(device, dtype, non_blocking)

# ============= BENCHMARKING FUNCTIONS =============

def benchmark_rms_norm(batch_size: int, seq_len: int, d_model: int, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark RMSNorm implementations"""
    device = torch.device('cuda')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    weight = torch.ones(d_model, device=device, dtype=torch.float16)
    
    # PyTorch RMSNorm
    pytorch_norm = nn.RMSNorm(d_model).to(device, dtype=torch.float16)
    
    # Triton RMSNorm
    triton_norm = TritonRMSNormLayer(d_model).to(device, dtype=torch.float16)
    
    # Extended warmup to ensure kernels are compiled
    print("    Warming up kernels...")
    for _ in range(20):
        _ = pytorch_norm(x)
        _ = triton_norm(x)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_norm(x)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_runs):
        _ = triton_norm(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    return {
        'pytorch_time': pytorch_time / num_runs * 1000,  # ms
        'triton_time': triton_time / num_runs * 1000,    # ms
        'speedup': pytorch_time / triton_time
    }

def benchmark_rotary(batch_size: int, seq_len: int, n_heads: int, d_head: int, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark Rotary implementations"""
    device = torch.device('cuda')
    
    # Create test data
    q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=torch.float16)
    
    # PyTorch Rotary
    pytorch_rotary = Rotary(d_head, seq_len).to(device)
    
    # Triton Rotary
    triton_rotary = TritonRotary(d_head, seq_len).to(device)
    
    # Warmup
    for _ in range(10):
        _ = pytorch_rotary(q)
        _ = triton_rotary(q, k)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_rotary(q)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_runs):
        _ = triton_rotary(q, k)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    return {
        'pytorch_time': pytorch_time / num_runs * 1000,  # ms
        'triton_time': triton_time / num_runs * 1000,    # ms
        'speedup': pytorch_time / triton_time
    }

def benchmark_mlp(batch_size: int, seq_len: int, d_model: int, d_ff: int, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark MLP implementations"""
    device = torch.device('cuda')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    
    # PyTorch FeedForward
    pytorch_mlp = FeedForward(d_model, d_ff).to(device, dtype=torch.float16)
    
    # Triton GatedMLP
    triton_mlp = TritonGatedMLP(d_model, d_ff).to(device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = pytorch_mlp(x)
        _ = triton_mlp(x)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_mlp(x)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_runs):
        _ = triton_mlp(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    return {
        'pytorch_time': pytorch_time / num_runs * 1000,  # ms
        'triton_time': triton_time / num_runs * 1000,    # ms
        'speedup': pytorch_time / triton_time
    }

def benchmark_newton_schulz(matrix_size: int, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark Newton-Schulz implementations"""
    device = torch.device('cuda')
    
    # Create test data
    G = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = zeropower_via_newtonschulz5(G)
        _ = zeropower_via_triton(G)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_runs):
        _ = zeropower_via_newtonschulz5(G)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_runs):
        _ = zeropower_via_triton(G)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    return {
        'pytorch_time': pytorch_time / num_runs * 1000,  # ms
        'triton_time': triton_time / num_runs * 1000,    # ms
        'speedup': pytorch_time / triton_time
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different configurations"""
    print("🚀 Triton vs PyTorch Performance Benchmark")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {'batch_size': 16, 'seq_len': 256, 'd_model': 384, 'n_heads': 8, 'd_ff': 1536},
        {'batch_size': 32, 'seq_len': 512, 'd_model': 768, 'n_heads': 12, 'd_ff': 3072},
        {'batch_size': 64, 'seq_len': 1024, 'd_model': 1024, 'n_heads': 16, 'd_ff': 4096},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n�� Configuration {i+1}: {config['batch_size']}x{config['seq_len']}x{config['d_model']}")
        print("-" * 50)
        
        try:
            # RMSNorm benchmark
            print("  Testing RMSNorm...")
            rms_results = benchmark_rms_norm(
                config['batch_size'], config['seq_len'], config['d_model']
            )
            print(f"  RMSNorm:")
            print(f"    PyTorch: {rms_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {rms_results['triton_time']:.3f} ms")
            print(f"    Speedup: {rms_results['speedup']:.2f}x")
            
            # Rotary benchmark
            print("  Testing Rotary...")
            rotary_results = benchmark_rotary(
                config['batch_size'], config['seq_len'], config['n_heads'], config['d_model'] // config['n_heads']
            )
            print(f"  Rotary:")
            print(f"    PyTorch: {rotary_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {rotary_results['triton_time']:.3f} ms")
            print(f"    Speedup: {rotary_results['speedup']:.2f}x")
            
            # MLP benchmark
            print("  Testing MLP...")
            mlp_results = benchmark_mlp(
                config['batch_size'], config['seq_len'], config['d_model'], config['d_ff']
            )
            print(f"  MLP:")
            print(f"    PyTorch: {mlp_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {mlp_results['triton_time']:.3f} ms")
            print(f"    Speedup: {mlp_results['speedup']:.2f}x")
            
        except Exception as e:
            print(f"  ❌ Error in configuration {i+1}: {e}")
            continue
    
    # Newton-Schulz benchmark
    print(f"\n�� Newton-Schulz Orthogonalization")
    print("-" * 50)
    for matrix_size in [128, 256, 512]:
        try:
            print(f"  Testing {matrix_size}x{matrix_size}...")
            ns_results = benchmark_newton_schulz(matrix_size)
            print(f"  Matrix {matrix_size}x{matrix_size}:")
            print(f"    PyTorch: {ns_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {ns_results['triton_time']:.3f} ms")
            print(f"    Speedup: {ns_results['speedup']:.2f}x")
        except Exception as e:
            print(f"  ❌ Error in Newton-Schulz {matrix_size}: {e}")
            continue
    
    print(f"\n✅ Benchmark completed!")

def run_memory_efficiency_test():
    """Test memory efficiency of Triton vs PyTorch"""
    print("\n💾 Memory Efficiency Test")
    print("=" * 40)
    
    device = torch.device('cuda')
    batch_size, seq_len, d_model = 32, 512, 768
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    
    # Measure memory before
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # PyTorch RMSNorm
    pytorch_norm = nn.RMSNorm(d_model).to(device, dtype=torch.float16)
    _ = pytorch_norm(x)
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Reset and measure Triton
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    triton_norm = TritonRMSNormLayer(d_model).to(device, dtype=torch.float16)
    _ = triton_norm(x)
    triton_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"Memory Usage:")
    print(f"  PyTorch: {pytorch_memory:.1f} MB")
    print(f"  Triton:  {triton_memory:.1f} MB")
    print(f"  Memory reduction: {((pytorch_memory - triton_memory) / pytorch_memory * 100):.1f}%")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires a GPU.")
        exit(1)
    
    print(f"🔧 CUDA Device: {torch.cuda.get_device_name()}")
    print(f"🔧 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Run comprehensive benchmark
        run_comprehensive_benchmark()
        
        # Run memory efficiency test
        run_memory_efficiency_test()
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()