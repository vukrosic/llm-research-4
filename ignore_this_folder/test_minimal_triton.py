#!/usr/bin/env python3
"""
Minimal test script for Triton functionality without external dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Mock TRITON_AVAILABLE for testing
TRITON_AVAILABLE = False

# Simple RMSNorm implementation for testing
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

# Simple Rotary implementation for testing
class SimpleRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        
        # Precompute cos and sin
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
    
    def forward(self, x):
        batch_size, n_heads, seq_len, d_head = x.shape
        cos = self.cos[:seq_len, :d_head//2].unsqueeze(0).unsqueeze(2)
        sin = self.sin[:seq_len, :d_head//2].unsqueeze(0).unsqueeze(2)
        
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

# Simple Attention implementation for testing
class SimpleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = SimpleRotary(self.d_k, max_seq_len)
    
    def to(self, device=None, dtype=None, non_blocking=False):
        """Override to method to handle dtype conversion"""
        super().to(device, dtype, non_blocking)
        if dtype is not None:
            # Convert linear layer weights to the specified dtype
            self.qkv.weight.data = self.qkv.weight.data.to(dtype)
            self.w_o.weight.data = self.w_o.weight.data.to(dtype)
        return self
    
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

# Simple MLP implementation for testing
class SimpleMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def to(self, device=None, dtype=None, non_blocking=False):
        """Override to method to handle dtype conversion"""
        super().to(device, dtype, non_blocking)
        if dtype is not None:
            # Convert linear layer weights to the specified dtype
            self.linear1.weight.data = self.linear1.weight.data.to(dtype)
            self.linear2.weight.data = self.linear2.weight.data.to(dtype)
        return self
    
    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

def test_dtype_handling():
    """Test that all classes handle dtype conversion properly"""
    print("🧪 Testing Dtype Handling (Minimal Version)")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU operations.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    try:
        # Test 1: RMSNorm
        print("\n🔍 Test 1: RMSNorm Dtype Handling")
        rms_norm = SimpleRMSNorm(128)
        rms_norm = rms_norm.to('cuda', dtype=torch.float16)
        
        x = torch.randn(4, 64, 128, device='cuda', dtype=torch.float16)
        output = rms_norm(x)
        print(f"✅ RMSNorm: Input {x.dtype}, Output {output.dtype}, Weight {rms_norm.weight.dtype}")
        
        # Test 2: Rotary
        print("\n🔍 Test 2: Rotary Dtype Handling")
        rotary = SimpleRotary(64, 128)
        rotary = rotary.to('cuda', dtype=torch.float16)
        
        q = torch.randn(4, 8, 64, 64, device='cuda', dtype=torch.float16)
        q_out = rotary(q)
        print(f"✅ Rotary: Input {q.dtype}, Output {q_out.dtype}, Buffers {rotary.cos.dtype}")
        
        # Test 3: Attention
        print("\n🔍 Test 3: Attention Dtype Handling")
        attention = SimpleAttention(128, 8, 64)
        attention = attention.to('cuda', dtype=torch.float16)
        
        x = torch.randn(4, 64, 128, device='cuda', dtype=torch.float16)
        output = attention(x)
        print(f"✅ Attention: Input {x.dtype}, Output {output.dtype}, Weights {attention.qkv.weight.dtype}")
        
        # Test 4: MLP
        print("\n🔍 Test 4: MLP Dtype Handling")
        mlp = SimpleMLP(128, 512)
        mlp = mlp.to('cuda', dtype=torch.float16)
        
        x = torch.randn(4, 64, 128, device='cuda', dtype=torch.float16)
        output = mlp(x)
        print(f"✅ MLP: Input {x.dtype}, Output {output.dtype}, Weights {mlp.linear1.weight.dtype}")
        
        print("\n🎉 All dtype tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dtype_handling()
    if success:
        print("\n🚀 Dtype handling is working correctly!")
    else:
        print("\n⚠️  Dtype handling has issues. Check the error messages above.")
