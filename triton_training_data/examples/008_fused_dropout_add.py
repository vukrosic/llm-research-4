"""Fused Dropout and Add operation."""

# <PYTHON>
import torch
import torch.nn.functional as F

def dropout_add(x, residual, p, is_training):
    # Two separate operations
    if is_training:
        dropped_x = F.dropout(x, p=p) # Kernel 1: dropout
    else:
        dropped_x = x
    output = dropped_x + residual     # Kernel 2: add
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def dropout_add_kernel(
    x_ptr, residual_ptr, output_ptr,
    n_elements,
    p, seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load x and residual
    x = tl.load(x_ptr + offset, mask=mask)
    residual = tl.load(residual_ptr + offset, mask=mask)

    # Generate dropout mask and apply it
    random = tl.rand(seed, offset)
    dropout_mask = random > p
    # Inverted dropout: scale by 1/ (1-p)
    scale = 1.0 / (1.0 - p)
    dropped_x = tl.where(dropout_mask, x * scale, 0.0)

    # Add residual
    output = dropped_x + residual

    # Store result
    tl.store(output_ptr + offset, output, mask=mask)

def dropout_add(x, residual, p, is_training):
    if not is_training or p == 0:
        return x + residual
        
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Seed needs to be different for each run
    seed = torch.randint(0, 10000, (1,)).item()

    dropout_add_kernel[grid](
        x, residual, output,
        n_elements, p, seed,
        BLOCK_SIZE=1024
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    size = 2048 * 2048
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    residual = torch.randn(size, device='cuda', dtype=torch.float32)
    p = 0.1
    is_training = True
    # Note: Triton and PyTorch dropout will not be numerically identical 
    # due to different random number generation. The purpose is to test
    # that the fused operation runs, not bit-for-bit correctness.
    # For a real test, one would check statistical properties.
    return (x, residual, p, is_training)
# </TEST>
