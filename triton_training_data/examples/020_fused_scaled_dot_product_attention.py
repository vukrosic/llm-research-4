"""Fused scaled dot-product attention"""

# <PYTHON>
import torch

def fused_scaled_dot_product_attention(q, k, v, scale_factor):
    # Multiple separate operations in PyTorch:
    # 1. Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1))  # Kernel 1: matmul
    
    # 2. Scale attention scores
    attn_scores = attn_scores * scale_factor            # Kernel 2: multiply
    
    # 3. Apply softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)   # Kernel 3: softmax
    
    # 4. Apply attention weights to values
    output = torch.matmul(attn_weights, v)              # Kernel 4: matmul
    
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def scaled_dot_product_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    seq_len, head_dim, num_heads,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ov,
    scale_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load Q block
    q_ptrs = q_ptr + (batch_id * stride_qb + head_id * stride_qh + 
                      offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
    
    # Compute QK^T (attention scores)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for n in range(0, seq_len, BLOCK_N):
        k_ptrs = k_ptr + (batch_id * stride_kb + head_id * stride_kh + 
                          (n + offs_n[:, None]) * stride_kn + offs_k[None, :] * stride_kk)
        k = tl.load(k_ptrs, mask=((n + offs_n[:, None]) < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
        
        # Compute dot product
        acc_block = tl.dot(q, tl.trans(k))
        acc = tl.where((offs_m[:, None] < seq_len) & ((n + offs_n[None, :]) < seq_len), 
                       acc + acc_block, acc)
    
    # Scale attention scores
    acc = acc * scale_factor
    
    # Apply softmax
    max_val = tl.max(acc, 1, keep_dims=True)
    acc = acc - max_val
    numerator = tl.exp(acc)
    denominator = tl.sum(numerator, 1, keep_dims=True)
    attn_weights = numerator / denominator
    
    # Apply attention weights to V
    output = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    for n in range(0, seq_len, BLOCK_N):
        v_ptrs = v_ptr + (batch_id * stride_vb + head_id * stride_vh + 
                          (n + offs_n[:, None]) * stride_vn + offs_k[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=((n + offs_n[:, None]) < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
        
        # Apply attention weights
        weights = tl.load(attn_weights + (offs_m[:, None] * seq_len + (n + offs_n[None, :])), 
                         mask=(offs_m[:, None] < seq_len) & ((n + offs_n[None, :]) < seq_len), other=0.0)
        output += tl.dot(weights, v)
    
    # Store result
    output_ptrs = output_ptr + (batch_id * stride_ob + head_id * stride_oh + 
                                offs_m[:, None] * stride_om + offs_k[None, :] * stride_ov)
    tl.store(output_ptrs, output, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim))

def fused_scaled_dot_product_attention(q, k, v, scale_factor):
    batch_size, num_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)
    
    grid = (batch_size, num_heads, triton.cdiv(seq_len, 32))
    
    scaled_dot_product_attention_kernel[grid](
        q, k, v, output,
        seq_len, head_dim, num_heads,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale_factor,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 4
    num_heads = 8
    seq_len = 128
    head_dim = 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    scale_factor = 0.125  # 1/sqrt(64)
    return (q, k, v, scale_factor)
# </TEST>