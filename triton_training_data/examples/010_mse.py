"""Fused mean squared error computation"""

# <PYTHON>
def mse(x, y):
    # Three separate kernels in PyTorch:
    diff = x - y              # Kernel 1: subtraction
    squared_diff = diff ** 2  # Kernel 2: power
    result = torch.mean(squared_diff)  # Kernel 3: mean reduction
    return result
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def mse_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single fused kernel computing mean squared error
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    
    # Compute squared difference
    diff = x - y
    squared_diff = diff * diff
    
    # Compute mean (sum all elements and divide by n_elements)
    sum_squared_diff = tl.sum(squared_diff)
    mean_squared_diff = sum_squared_diff / n_elements
    
    # Store result (only first thread in the block stores the result)
    if pid == 0:
        tl.store(output_ptr, mean_squared_diff)

def mse(x, y):
    # Wrapper that launches the fused kernel
    output = torch.empty(1, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mse_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    size = 1000000
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    return (x, y)
# </TEST>
