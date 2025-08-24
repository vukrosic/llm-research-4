"""Fused log softmax computation"""

# <PYTHON>
import torch.nn.functional as F

def log_softmax(x):
    # Two separate kernels in PyTorch:
    x = F.softmax(x, dim=-1)  # Kernel 1: softmax
    x = torch.log(x)          # Kernel 2: log
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl
import math

@triton.jit
def log_softmax_kernel(input_ptr, output_ptr, n_cols,
                       BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    
    # Load entire row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
    
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    
    # Compute softmax
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp
    
    # Compute log softmax directly
    log_softmax = x - tl.log(sum_exp)
    
    # Store
    tl.store(output_ptr + row_start + cols, log_softmax, mask=mask)

def log_softmax(x):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    log_softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 32
    seq_len = 512
    x = torch.randn(batch_size, seq_len, device='cuda')
    return (x,)
# </TEST>
