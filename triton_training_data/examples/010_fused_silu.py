"""Fused SiLU (Swish) activation function: x * sigmoid(x)"""

# <PYTHON>
import torch

def fused_silu(x):
    # Two separate kernels in PyTorch:
    sigmoid_x = torch.sigmoid(x)  # Kernel 1: sigmoid
    output = x * sigmoid_x        # Kernel 2: multiply
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute SiLU in one step: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    output = x * sigmoid_x
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_silu(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
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