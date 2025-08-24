"""Scaled softmax for attention: softmax(x / sqrt(scale))"""

# <PYTHON>
def scaled_softmax(x, scale):
    # Multiple kernels:
    x_scaled = x / math.sqrt(scale)  # Kernel 1: div
    x_max = x_scaled.max(dim=-1, keepdim=True)[0]   # Kernel 2: max reduction
    x_safe = x_scaled - x_max                             # Kernel 3: sub
    exp_x = torch.exp(x_safe)                      # Kernel 4: exp
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
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_cols
    
    # Load entire row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Scale
    scale_sqrt = tl.sqrt(scale.to(tl.float32)) # Ensure scale is float for sqrt
    row = row / scale_sqrt
    
    # Softmax - all fused
    row_max = tl.max(row, axis=0)
    row = row - row_max
    exp_row = tl.exp(row)
    sum_exp_row = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_exp_row
    
    # Store
    output_row_start_ptr = output_ptr + row_idx * n_cols
    tl.store(output_row_start_ptr + col_offsets, softmax_row, mask=mask)

def scaled_softmax(x, scale):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    # Let triton select next power of 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    scaled_softmax_kernel[grid](x, output, n_cols, scale, BLOCK_SIZE=BLOCK_SIZE)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 32
    seq_len = 512
    x = torch.randn(batch_size, seq_len, device='cuda', dtype=torch.float32)
    scale = torch.tensor(64.0, device='cuda', dtype=torch.float32)
    return (x, scale)
# </TEST>