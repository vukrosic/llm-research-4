"""Linear transformation with fused ReLU activation"""

# <PYTHON>
def linear_relu(x, weight, bias):
    # Three separate kernels in PyTorch:
    output = torch.matmul(x, weight)  # Kernel 1: matmul
    output = output + bias            # Kernel 2: add
    output = torch.relu(output)       # Kernel 3: relu
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def linear_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk, # Note: Swapped in dot, so wk, wn
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_chunk = tl.load(x_ptrs)
        w_chunk = tl.load(w_ptrs)
        acc += tl.dot(x_chunk, w_chunk)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias_ptrs = bias_ptr + offs_n
    bias_chunk = tl.load(bias_ptrs)
    acc = acc + bias_chunk[None, :]

    # Apply ReLU
    output = tl.maximum(acc, 0)

    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(output_ptrs, output)

def linear_relu(x, weight, bias):
    M, K = x.shape
    K, N = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    linear_relu_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 128
    in_features = 768
    out_features = 3072
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    weight = torch.randn(in_features, out_features, device='cuda', dtype=torch.float16)
    bias = torch.randn(out_features, device='cuda', dtype=torch.float16)
    return (x, weight, bias)
# </TEST>