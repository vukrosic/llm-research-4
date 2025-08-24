"""Fused attention score computation: QK^T/sqrt(d) + mask -> softmax"""

# <PYTHON>
import torch

def fused_attention_scores(q, k, mask, scale):
    # Multiple separate kernels in PyTorch:
    scores = torch.matmul(q, k.transpose(-2, -1))  # Kernel 1: matmul
    scores = scores * scale                        # Kernel 2: scale
    scores = scores + mask                         # Kernel 3: add mask
    scores = torch.softmax(scores, dim=-1)         # Kernel 4: softmax
    return scores
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, mask_ptr, output_ptr,
    seq_len, head_dim, num_heads,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_on,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load Q block
    q_ptrs = q_ptr + (batch_id * stride_qb + head_id * stride_qh + 
                      offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
    
    # Compute QK^T
    for n in range(0, seq_len, BLOCK_N):
        k_ptrs = k_ptr + (batch_id * stride_kb + head_id * stride_kh + 
                          (n + offs_n[:, None]) * stride_kn + offs_k[None, :] * stride_kk)
        k = tl.load(k_ptrs, mask=((n + offs_n[:, None]) < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
        
        # Compute dot product
        acc_block = tl.dot(q, tl.trans(k))
        acc = tl.where((offs_m[:, None] < seq_len) & ((n + offs_n[None, :]) < seq_len), 
                       acc + acc_block, acc)
    
    # Scale
    acc = acc * scale
    
    # Add mask
    mask_ptrs = mask_ptr + (batch_id * stride_mb + head_id * stride_mh + 
                            offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn)
    mask_vals = tl.load(mask_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len), other=0.0)
    acc = acc + mask_vals
    
    # Softmax
    max_val = tl.max(acc, 1, keep_dims=True)
    acc = acc - max_val
    numerator = tl.exp(acc)
    denominator = tl.sum(numerator, 1, keep_dims=True)
    softmax_output = numerator / denominator
    
    # Store result
    output_ptrs = output_ptr + (batch_id * stride_ob + head_id * stride_oh + 
                                offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(output_ptrs, softmax_output, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len))

def fused_attention_scores(q, k, mask, scale):
    batch_size, num_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)
    
    grid = (batch_size, num_heads, triton.cdiv(seq_len, 32))
    
    attention_scores_kernel[grid](
        q, k, mask, output,
        seq_len, head_dim, num_heads,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
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
    mask = torch.randn(batch_size, num_heads, seq_len, seq_len, device='cuda', dtype=torch.float16)
    scale = 0.125  # 1/sqrt(64)
    return (q, k, mask, scale)
# </TEST>