"""Fused residual connection and layer normalization"""

# <PYTHON>
def residual_layernorm(x, residual):
    # Two separate kernels in PyTorch:
    x = x + residual              # Kernel 1: addition
    x = torch.nn.functional.layer_norm(x, x.shape[-1:])  # Kernel 2: layer norm
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def residual_layernorm_kernel(
    x_ptr, residual_ptr, output_ptr, 
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Single fused kernel doing residual connection + layer norm
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    # Load inputs
    x = tl.load(x_ptr + offset, mask=mask)
    residual = tl.load(residual_ptr + offset, mask=mask)
    
    # Add residual connection
    x = x + residual
    
    # Compute layer norm statistics
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) * (x - mean)) / N
    rstd = tl.rsqrt(var + eps)
    
    # Normalize
    output = (x - mean) * rstd
    
    # Store result
    tl.store(output_ptr + offset, output, mask=mask)

def residual_layernorm(x, residual):
    # Wrapper that launches the fused kernel
    output = torch.empty_like(x)
    N = x.numel()
    eps = 1e-5
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    residual_layernorm_kernel[grid](x, residual, output, N, eps, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 32
    seq_len = 512
    hidden_size = 768
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    residual = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    return (x, residual)
# </TEST>
