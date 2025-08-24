"""Fused Gated Linear Unit (GLU)."""

# <PYTHON>
import torch

def gated_linear_unit(x, w, v):
    # Three separate kernels
    gate = x @ v             # Kernel 1: Matmul
    activation = torch.sigmoid(gate) # Kernel 2: Sigmoid
    y = x @ w                # Kernel 3: Matmul
    return y * activation    # Kernel 4: Element-wise multiply
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def glu_kernel(
    x_ptr, w_ptr, v_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_vk, stride_vn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc_w = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        v_ptrs = v_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)
        
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0)
        
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0)

        v_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        v_chunk = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        acc_w += tl.dot(x_chunk, w_chunk)
        acc_v += tl.dot(x_chunk, v_chunk)
    
    # Fused sigmoid and multiply
    gate = tl.sigmoid(acc_v)
    output = acc_w * gate

    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output.to(x_ptr.dtype.element_ty), mask=out_mask)

def gated_linear_unit(x, w, v):
    M, K = x.shape
    K, N = w.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    glu_kernel[grid](
        x, w, v, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        v.stride(0), v.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch, seq_len, dim_in, dim_out = 16, 512, 768, 3072
    x = torch.randn(batch * seq_len, dim_in, device='cuda', dtype=torch.float16)
    w = torch.randn(dim_in, dim_out, device='cuda', dtype=torch.float16)
    v = torch.randn(dim_in, dim_out, device='cuda', dtype=torch.float16)
    return (x, w, v)
# </TEST>
