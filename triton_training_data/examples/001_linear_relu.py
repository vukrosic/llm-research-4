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
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Single fused kernel doing matmul + bias + relu
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute one block of output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load blocks
        a = tl.load(x_ptr + pid_m * BLOCK_M * stride_xk + tl.arange(0, BLOCK_K) * stride_xk + tl.arange(0, BLOCK_M)[:, None] * stride_xm)
        b = tl.load(weight_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_wk + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) * stride_wn)
        acc += tl.dot(a, b)
    
    # Add bias and apply ReLU in-place
    bias = tl.load(bias_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    output = acc + bias[None, :]
    output = tl.maximum(output, 0)  # Fused ReLU
    
    # Store result
    tl.store(output_ptr + pid_m * BLOCK_M * N + pid_n * BLOCK_N + tl.arange(0, BLOCK_M)[:, None] * N + tl.arange(0, BLOCK_N)[None, :], output)

def linear_relu(x, weight, bias):
    # Wrapper that launches the fused kernel
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
