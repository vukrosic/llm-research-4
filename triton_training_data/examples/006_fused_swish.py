"""Fused Swish (or SiLU) activation function."""

# <PYTHON>
import torch

def fused_swish(x):
    # Two separate kernels
    sigmoid_x = torch.sigmoid(x) # Kernel 1: sigmoid
    output = x * sigmoid_x       # Kernel 2: multiply
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def swish_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask)
    
    # Fused computation: x * sigmoid(x)
    sigmoid_x = 1 / (1 + tl.exp(-x))
    output = x * sigmoid_x
    
    tl.store(output_ptr + offset, output, mask=mask)

def fused_swish(x):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    swish_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    size = 4096 * 4096
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    return (x,)
# </TEST>
