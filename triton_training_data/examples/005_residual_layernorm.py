"""Fused LayerNorm with a residual connection."""

# <PYTHON>
import torch
import torch.nn.functional as F

def residual_layernorm(x, residual, gamma, beta, eps):
    # Two separate operations
    y = x + residual  # Kernel 1: add
    # Kernel 2: LayerNorm (which is itself multiple kernels)
    return F.layer_norm(y, (y.shape[-1],), weight=gamma, bias=beta, eps=eps)
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    x_ptr, residual_ptr, gamma_ptr, beta_ptr, output_ptr,
    n_cols,
    eps,
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

    # Compute mean and variance
    mean = tl.sum(y, axis=0) / n_cols
    y_minus_mean = y - mean
    var = tl.sum(y_minus_mean * y_minus_mean, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply scale/shift
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    normalized = (y - mean) * inv_std
    output = normalized * gamma + beta

    # Store result
    tl.store(row_out_ptr + col_offsets, output, mask=mask)

def residual_layernorm(x, residual, gamma, beta, eps):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Let Triton pick the best block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    layernorm_kernel[grid](
        x, residual, gamma, beta, output,
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch, seq_len, dim = 32, 512, 768
    x = torch.randn(batch * seq_len, dim, device='cuda', dtype=torch.float16)
    residual = torch.randn(batch * seq_len, dim, device='cuda', dtype=torch.float16)
    gamma = torch.randn(dim, device='cuda', dtype=torch.float16)
    beta = torch.randn(dim, device='cuda', dtype=torch.float16)
    eps = 1e-5
    return (x, residual, gamma, beta, eps)
# </TEST>
