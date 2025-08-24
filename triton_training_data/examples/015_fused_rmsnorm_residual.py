"""Fused RMSNorm with residual connection"""

# <PYTHON>
import torch

def fused_rmsnorm_residual(x, residual, weight, eps=1e-6):
    # Multiple separate operations in PyTorch:
    x = x + residual                    # Kernel 1: add residual
    variance = x.pow(2).mean(-1, keepdim=True)  # Kernel 2: square and mean
    x = x * torch.rsqrt(variance + eps) # Kernel 3: rsqrt and multiply
    x = x * weight                      # Kernel 4: scale
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    row_x_ptr = x_ptr + row_idx * n_cols
    row_res_ptr = residual_ptr + row_idx * n_cols
    row_out_ptr = output_ptr + row_idx * n_cols

    # Load row, add residual
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x = tl.load(row_x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(row_res_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x + residual

    # Compute RMSNorm
    square_sum = tl.sum(y * y, axis=0)
    mean_square = square_sum / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)
    normalized = y * inv_rms

    # Apply weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = normalized * weight

    # Store result
    tl.store(row_out_ptr + col_offsets, output, mask=mask)

def fused_rmsnorm_residual(x, residual, weight, eps=1e-6):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Let Triton pick the best block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    rmsnorm_residual_kernel[grid](
        x, residual, weight, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch, seq_len, dim = 8, 512, 768
    x = torch.randn(batch * seq_len, dim, device='cuda', dtype=torch.float16)
    residual = torch.randn(batch * seq_len, dim, device='cuda', dtype=torch.float16)
    weight = torch.randn(dim, device='cuda', dtype=torch.float16)
    eps = 1e-6
    return (x, residual, weight, eps)
# </TEST>