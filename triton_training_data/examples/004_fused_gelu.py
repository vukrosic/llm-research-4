"""Fused GELU activation function."""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_gelu(x):
    # PyTorch's GELU is often a single optimized kernel, 
    # but conceptually it's a series of operations.
    # Here we use the approximate version for demonstration.
    return F.gelu(x, approximate='tanh')
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using the tanh approximation for GELU
    cdf = 0.5 * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tl.pow(x, 3))))
    output = x * cdf
    
    tl.store(output_ptr + offset, output, mask=mask)

def fused_gelu(x):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    size = 2048 * 2048
    x = torch.randn(size, device='cuda', dtype=torch.float32) * 5
    return (x,)
# </TEST>
