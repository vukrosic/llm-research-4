"""Fused derivative computation: x * (1 - x)"""

# <PYTHON>
def derivative(x):
    # Two separate kernels in PyTorch:
    complement = 1 - x    # Kernel 1: subtraction
    result = x * complement  # Kernel 2: multiplication
    return result
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def derivative_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load input
    x = tl.load(x_ptr + offset, mask=mask)
    
    # Compute derivative: x * (1 - x)
    complement = 1 - x
    output = x * complement
    
    # Store result
    tl.store(output_ptr + offset, output, mask=mask)

def derivative(x):
    # Wrapper that launches the fused kernel
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    derivative_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    size = 1000000
    x = torch.randn(size, device='cuda')
    return (x,)
# </TEST>
