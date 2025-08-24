"""Fused dropout with ReLU activation"""

# <PYTHON>
import torch.nn.functional as F

def dropout_relu(x, weight, bias, p=0.1):
    # Four separate kernels in PyTorch:
    output = torch.matmul(x, weight)  # Kernel 1: matmul
    output = output + bias            # Kernel 2: add bias
    output = F.relu(output)          # Kernel 3: ReLU activation
    output = F.dropout(output, p)    # Kernel 4: dropout
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def dropout_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K, p,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Single fused kernel doing matmul + bias + ReLU + dropout
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute one block of output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load blocks
        a = tl.load(x_ptr + ...)
        b = tl.load(weight_ptr + ...)
        acc += tl.dot(a, b)
    
    # Add bias
    bias = tl.load(bias_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    output = acc + bias[None, :]
    
    # Apply ReLU activation
    output = tl.maximum(output, 0)
    
    # Apply dropout (simplified implementation)
    # In practice, you would use a proper random number generator
    output = tl.where(tl.rand(seed=0) > p, output / (1 - p), 0)
    
    # Store result
    tl.store(output_ptr + ..., output)

def dropout_relu(x, weight, bias, p=0.1):
    # Wrapper that launches the fused kernel
    M, K = x.shape
    K, N = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    dropout_relu_kernel[grid](
        x, weight, bias, output,
        M, N, K, p,
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
