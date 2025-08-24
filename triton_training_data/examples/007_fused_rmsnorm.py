"""Fused RMS Normalization."""

# <PYTHON>
import torch

def residual_rmsnorm(x, weight, eps):
    # RMSNorm is variance-only, no mean subtraction
    # 1. Calculate variance (mean of squares)
    variance = x.pow(2).mean(-1, keepdim=True)
    # 2. Normalize
    hidden_states = x * torch.rsqrt(variance + eps)
    # 3. Apply scaling weight
    return weight * hidden_states
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    row_x_ptr = x_ptr + row_idx * n_cols
    row_out_ptr = output_ptr + row_idx * n_cols

    # Compute variance
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(row_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize and apply weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask)
    normalized = x * rstd
    output = normalized * weight

    # Store result
    tl.store(row_out_ptr + col_offsets, output, mask=mask)

def residual_rmsnorm(x, weight, eps):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    rmsnorm_kernel[grid](
        x, weight, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch, dim = 128, 768
    x = torch.randn(batch, dim, device='cuda', dtype=torch.float16)
    weight = torch.randn(dim, device='cuda', dtype=torch.float16)
    eps = 1e-6
    return (x, weight, eps)
# </TEST>
