"""Fused Linear, ReLU, and Dropout."""

# <PYTHON>
import torch
import torch.nn.functional as F

def linear_relu_dropout(x, weight, bias, p, is_training):
    # 3 separate operations
    y = F.linear(x, weight, bias) # 1. linear
    y = F.relu(y)                 # 2. relu
    y = F.dropout(y, p, is_training) # 3. dropout
    return y
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def linear_relu_dropout_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    p, seed,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_chunk, w_chunk)

    # Add bias and apply ReLU
    bias_ptrs = bias_ptr + offs_n
    bias_chunk = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)
    output = acc + bias_chunk[None, :]
    output = tl.maximum(output, 0)

    # Apply dropout
    if p > 0.0:
        random = tl.rand(seed, tl.arange(0, BLOCK_M)[:, None] * N + tl.arange(0, BLOCK_N)[None, :])
        dropout_mask = random > p
        scale = 1.0 / (1.0 - p)
        output = tl.where(dropout_mask, output * scale, 0.0)

    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output.to(x_ptr.dtype.element_ty), mask=out_mask)

def linear_relu_dropout(x, weight, bias, p, is_training):
    if not is_training:
        # In inference, dropout is identity
        return torch.relu(F.linear(x, weight, bias))

    M, K = x.shape
    K, N = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    seed = torch.randint(0, 10000, (1,)).item()

    linear_relu_dropout_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        p, seed,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch, in_feat, out_feat = 128, 768, 3072
    x = torch.randn(batch, in_feat, device='cuda', dtype=torch.float16)
    weight = torch.randn(out_feat, in_feat, device='cuda', dtype=torch.float16)
    bias = torch.randn(out_feat, device='cuda', dtype=torch.float16)
    p = 0.1
    is_training = True
    # Note: As with dropout, results are not bit-identical.
    return (x, weight, bias, p, is_training)
# </TEST>