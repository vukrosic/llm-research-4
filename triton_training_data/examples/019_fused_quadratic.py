"""Fused quadratic activation: x + x^2"""

# <PYTHON>
import torch

def fused_quadratic(x):
    # Multiple separate kernels in PyTorch:
    x_squared = x * x        # Kernel 1: multiply
    output = x + x_squared   # Kernel 2: add
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def quadratic_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute quadratic: x + x^2
    x_squared = x * x
    output = x + x_squared
    
    # Store result
    tl.store(output_ptr + offsets, output.to(x.dtype), mask=mask)

def fused_quadratic(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    quadratic_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 128
    features = 3072
    x = torch.randn(batch_size, features, device='cuda', dtype=torch.float16)
    return (x,)
# </TEST>