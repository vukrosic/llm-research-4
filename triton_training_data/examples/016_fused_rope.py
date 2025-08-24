"""Fused rotary positional embedding (RoPE)"""

# <PYTHON>
import torch

def fused_rope(x, cos, sin):
    # Multiple operations in PyTorch:
    # Split x into even and odd parts
    x1 = x[..., ::2]        # Kernel 1: slice
    x2 = x[..., 1::2]       # Kernel 2: slice
    
    # Apply rotation
    y1 = x1 * cos - x2 * sin  # Kernel 3: multiply and subtract
    y2 = x1 * sin + x2 * cos  # Kernel 4: multiply and add
    
    # Interleave results
    output = torch.empty_like(x)
    output[..., ::2] = y1     # Kernel 5: slice assignment
    output[..., 1::2] = y2    # Kernel 6: slice assignment
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, output_ptr,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # For RoPE, we work with pairs of elements
    pair_offsets = offsets // 2
    is_odd = offsets % 2
    
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    cos = tl.load(cos_ptr + pair_offsets, mask=(pair_offsets * 2 < n_elements), other=0.0)
    sin = tl.load(sin_ptr + pair_offsets, mask=(pair_offsets * 2 < n_elements), other=0.0)
    
    # Apply rotation
    # For even indices: y = x * cos - x_shifted * sin
    # For odd indices: y = x_shifted * cos + x * sin
    x_shifted = tl.load(x_ptr + offsets + tl.where(is_odd, -1, 1), mask=mask, other=0.0)
    
    # Apply the rotation formula
    rotated = tl.where(
        is_odd,
        -x_shifted * cos + x * sin,  # For odd indices
        x * cos - x_shifted * sin    # For even indices
    )
    
    # Store result
    tl.store(output_ptr + offsets, rotated, mask=mask)

def fused_rope(x, cos, sin):
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = (triton.cdiv(n_elements, 1024),)
    
    rope_kernel[grid](
        x, cos, sin, output,
        n_elements,
        BLOCK_SIZE=1024
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 4
    seq_len = 512
    head_dim = 64
    num_heads = 12
    
    x = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    # Create position embeddings
    position = torch.arange(seq_len, device='cuda', dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda', dtype=torch.float32) / head_dim))
    freqs = position[:, None] * inv_freq[None, :]
    cos = torch.cos(freqs).repeat(1, 2).unsqueeze(0).unsqueeze(0)  # Add batch and head dims
    sin = torch.sin(freqs).repeat(1, 2).unsqueeze(0).unsqueeze(0)
    
    # Expand to match x shape
    cos = cos.expand_as(x).contiguous()
    sin = sin.expand_as(x).contiguous()
    
    return (x, cos, sin)
# </TEST>