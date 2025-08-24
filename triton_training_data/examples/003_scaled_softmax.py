"""Scaled softmax for attention: softmax(x / sqrt(scale))"""

# <PYTHON>
def scaled_softmax(x, scale):
    # Multiple kernels:
    x = x / torch.sqrt(torch.tensor(scale))  # Kernel 1: div
    x_max = x.max(dim=-1, keepdim=True)[0]   # Kernel 2: max reduction
    x = x - x_max                             # Kernel 3: sub
    exp_x = torch.exp(x)                      # Kernel 4: exp
    sum_exp = exp_x.sum(dim=-1, keepdim=True) # Kernel 5: sum reduction
    return exp_x / sum_exp                     # Kernel 6: div
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl
import math

@triton.jit
def scaled_softmax_kernel(input_ptr, output_ptr, n_cols, scale,
                         BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    
    # Load entire row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
    
    # Scale
    x = x / tl.sqrt(scale)
    
    # Softmax - all fused
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp
    
    # Store
    tl.store(output_ptr + row_start + cols, softmax, mask=mask)

def scaled_softmax(x, scale):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    scaled_softmax_kernel[grid](x, output, n_cols, float(scale), BLOCK_SIZE)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 32
    seq_len = 512
    x = torch.randn(batch_size, seq_len, device='cuda')
    scale = 64.0  # sqrt(d_head) for attention
    return (x, scale)
# </TEST>
