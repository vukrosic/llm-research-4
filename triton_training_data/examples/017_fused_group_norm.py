"""Fused group normalization"""

# <PYTHON>
import torch

def fused_group_norm(x, weight, bias, num_groups, eps=1e-5):
    # Multiple operations in PyTorch:
    batch, channels, height, width = x.shape
    
    # Reshape for group norm
    x = x.view(batch, num_groups, channels // num_groups, height, width)  # Kernel 1: view
    
    # Compute mean and variance for each group
    mean = x.mean(dim=(2, 3, 4), keepdim=True)   # Kernel 2: mean
    var = x.var(dim=(2, 3, 4), keepdim=True)     # Kernel 3: var
    x = (x - mean) / torch.sqrt(var + eps)       # Kernel 4: subtract, add, sqrt, divide
    
    # Reshape back
    x = x.view(batch, channels, height, width)   # Kernel 5: view
    
    # Apply weight and bias
    x = x * weight[None, :, None, None]          # Kernel 6: multiply
    x = x + bias[None, :, None, None]            # Kernel 7: add
    
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def group_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, channels, height, width, num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each program instance handles one group of one batch
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    # Calculate dimensions
    channels_per_group = channels // num_groups
    elements_per_group = channels_per_group * height * width
    
    # Calculate base offsets
    group_offset = batch_id * channels * height * width + group_id * elements_per_group
    weight_offset = group_id * channels_per_group
    bias_offset = group_id * channels_per_group
    
    # Process elements in this group
    for i in range(0, elements_per_group, BLOCK_SIZE):
        # Calculate offsets
        elem_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = elem_offsets < elements_per_group
        
        # Load data
        x_vals = tl.load(x_ptr + group_offset + elem_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Compute mean
        mean = tl.sum(x_vals, axis=0) / elements_per_group
        
        # Compute variance
        diff = x_vals - mean
        var = tl.sum(diff * diff, axis=0) / elements_per_group
        
        # Normalize
        normalized = (x_vals - mean) * tl.rsqrt(var + eps)
        
        # Load weight and bias for this channel
        channel_idx = (group_id * channels_per_group) + (elem_offsets // (height * width)) % channels_per_group
        weight_vals = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
        bias_vals = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
        
        # Apply weight and bias
        output_vals = normalized * weight_vals + bias_vals
        
        # Store result
        tl.store(output_ptr + group_offset + elem_offsets, output_vals, mask=mask)

def fused_group_norm(x, weight, bias, num_groups, eps=1e-5):
    batch, channels, height, width = x.shape
    output = torch.empty_like(x)
    
    # Grid: one block per group per batch
    grid = (batch, num_groups)
    
    group_norm_kernel[grid](
        x, weight, bias, output,
        batch, channels, height, width, num_groups,
        eps,
        BLOCK_SIZE=1024
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 4
    channels = 256
    height = 32
    width = 32
    num_groups = 32
    
    x = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float16)
    weight = torch.randn(channels, device='cuda', dtype=torch.float16)
    bias = torch.randn(channels, device='cuda', dtype=torch.float16)
    
    return (x, weight, bias, num_groups, 1e-5)
# </TEST>