"""Fused residual connection with dropout"""

# <PYTHON>
import torch.nn.functional as F

def residual_dropout(x, residual, p=0.1):
    # Two separate kernels in PyTorch:
    x = x + residual          # Kernel 1: addition
    x = F.dropout(x, p)      # Kernel 2: dropout
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def residual_dropout_kernel(
    x_ptr, residual_ptr, output_ptr, 
    N, p,
    BLOCK_SIZE: tl.constexpr
):
    # Single fused kernel doing residual connection + dropout
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    # Load inputs
    x = tl.load(x_ptr + offset, mask=mask)
    residual = tl.load(residual_ptr + offset, mask=mask)
    
    # Add residual connection
    x = x + residual
    
    # Apply dropout
    # In practice, you would use a proper random number generator
    x = tl.where(tl.rand(seed=0) > p, x / (1 - p), 0)
    
    # Store result
    tl.store(output_ptr + offset, x, mask=mask)

def residual_dropout(x, residual, p=0.1):
    # Wrapper that launches the fused kernel
    output = torch.empty_like(x)
    N = x.numel()
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    residual_dropout_kernel[grid](x, residual, output, N, p, BLOCK_SIZE=1024)
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
