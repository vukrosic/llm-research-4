"""Fused arithmetic operations: (x + y) * z + w"""

# <PYTHON>
def add_mul_add(x, y, z, w):
    # Three separate kernels:
    temp1 = x + y      # Kernel 1
    temp2 = temp1 * z  # Kernel 2  
    output = temp2 + w # Kernel 3
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def add_mul_add_kernel(x_ptr, y_ptr, z_ptr, w_ptr, output_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load all inputs once
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    z = tl.load(z_ptr + offset, mask=mask)
    w = tl.load(w_ptr + offset, mask=mask)
    
    # Fused computation - all in registers
    output = (x + y) * z + w
    
    # Single store
    tl.store(output_ptr + offset, output, mask=mask)

def add_mul_add(x, y, z, w):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_mul_add_kernel[grid](x, y, z, w, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    size = 1000000
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    z = torch.randn(size, device='cuda', dtype=torch.float32)
    w = torch.randn(size, device='cuda', dtype=torch.float32)
    return (x, y, z, w)
# </TEST>