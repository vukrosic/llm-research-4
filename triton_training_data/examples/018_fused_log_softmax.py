"""Fused log softmax"""

# <PYTHON>
import torch

def fused_log_softmax(x, dim=-1):
    # Multiple operations in PyTorch:
    x_max = x.max(dim=dim, keepdim=True)[0]   # Kernel 1: max reduction
    x = x - x_max                             # Kernel 2: subtract
    exp_x = torch.exp(x)                      # Kernel 3: exp
    sum_exp = exp_x.sum(dim=dim, keepdim=True) # Kernel 4: sum reduction
    log_sum_exp = torch.log(sum_exp)          # Kernel 5: log
    return x - log_sum_exp                     # Kernel 6: subtract
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(input_ptr, output_ptr, n_cols, n_rows,
                      BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    row_start = row_idx * n_cols
    input_row_ptr = input_ptr + row_start
    output_row_ptr = output_ptr + row_start

    # Load data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(input_row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    
    # Compute log softmax in one go
    # 1. Find max
    x_max = tl.max(x, axis=0)
    
    # 2. Subtract max for numerical stability
    x_minus_max = x - x_max
    
    # 3. Compute exp
    exp_x = tl.exp(x_minus_max)
    
    # 4. Sum exp
    sum_exp = tl.sum(exp_x, axis=0)
    
    # 5. Compute log of sum exp
    log_sum_exp = tl.log(sum_exp)
    
    # 6. Final result
    log_softmax_output = x_minus_max - log_sum_exp
    
    # Store result
    tl.store(output_row_ptr + col_offsets, log_softmax_output, mask=mask)

def fused_log_softmax(x, dim=-1):
    # For simplicity, assume dim=-1 and x is 2D
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    log_softmax_kernel[grid](
        x, output, n_cols, n_rows,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 32
    vocab_size = 32000
    x = torch.randn(batch_size, vocab_size, device='cuda', dtype=torch.float16)
    return (x,)
# </TEST>