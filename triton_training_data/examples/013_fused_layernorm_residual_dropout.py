"""Fused LayerNorm with residual connection and dropout"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_layernorm_residual_dropout(x, residual, gamma, beta, dropout_prob, training=True):
    # Multiple separate operations in PyTorch:
    x = x + residual                         # Kernel 1: add residual
    x = F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=1e-5)  # Kernel 2: layer norm
    x = F.dropout(x, p=dropout_prob, training=training)  # Kernel 3: dropout
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_residual_dropout_kernel(
    x_ptr, residual_ptr, gamma_ptr, beta_ptr, output_ptr,
    n_cols, eps, dropout_prob, seed,
    BLOCK_SIZE: tl.constexpr,
    TRAINING: tl.constexpr
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

    # Compute mean and variance for LayerNorm
    mean = tl.sum(y, axis=0) / n_cols
    y_minus_mean = y - mean
    var = tl.sum(y_minus_mean * y_minus_mean, axis=0) / n_cols
    inv_std = tl.rsqrt(var + eps)

    # Normalize and apply scale/shift
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    normalized = (y - mean) * inv_std
    output = normalized * gamma + beta

    # Apply dropout if training
    if TRAINING and dropout_prob > 0.0:
        # Generate random numbers using the row index and column offsets as seeds
        random = tl.rand(seed, row_idx * n_cols + col_offsets)
        dropout_mask = random > dropout_prob
        output = tl.where(dropout_mask, output / (1.0 - dropout_prob), 0.0)

    # Store result
    tl.store(row_out_ptr + col_offsets, output, mask=mask)

def fused_layernorm_residual_dropout(x, residual, gamma, beta, dropout_prob, training=True):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Let Triton pick the best block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    layernorm_residual_dropout_kernel[grid](
        x, residual, gamma, beta, output,
        n_cols, 1e-5, dropout_prob, 42,  # 42 as a fixed seed for reproducibility
        BLOCK_SIZE=BLOCK_SIZE,
        TRAINING=training
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch, seq_len, dim = 8, 512, 768
    x = torch.randn(batch * seq_len, dim, device='cuda', dtype=torch.float16)
    residual = torch.randn(batch * seq_len, dim, device='cuda', dtype=torch.float16)
    gamma = torch.randn(dim, device='cuda', dtype=torch.float16)
    beta = torch.randn(dim, device='cuda', dtype=torch.float16)
    dropout_prob = 0.1
    return (x, residual, gamma, beta, dropout_prob, True)
# </TEST>