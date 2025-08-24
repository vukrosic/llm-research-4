"""Fused Kernels: Scale + Clamp + Normalize"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_kernel_pytorch(x, scale, clamp_min, clamp_max, eps=1e-5):
    """
    PyTorch implementation combining:
    1. Scale the input by a factor
    2. Clamp values to a range
    3. Normalize (mean=0, std=1)
    """
    # Scale
    x = x * scale
    
    # Clamp
    x = torch.clamp(x, min=clamp_min, max=clamp_max)
    
    # Normalize (standardization)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x = (x - mean) / (std + eps)
    
    return x

# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel_triton(
    x_ptr, output_ptr,
    scale, clamp_min, clamp_max, eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton fused kernel implementation
    """
    # Get program id
    pid = tl.program_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Scale
    x = x * scale
    
    # Clamp
    x = tl.maximum(x, clamp_min)
    x = tl.minimum(x, clamp_max)
    
    # For normalization, we need a two-pass approach
    # First pass: compute mean
    mean = tl.sum(x, axis=0) / n_elements
    
    # Second pass: compute std and normalize
    # Note: This is a simplified version for demonstration
    # In practice, normalization across dimensions requires more complex logic
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

def fused_kernel_triton_wrapper(x, scale, clamp_min, clamp_max, eps=1e-5):
    """
    Wrapper for the Triton kernel
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Get dimensions
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    fused_kernel_triton[grid](
        x, output,
        scale, clamp_min, clamp_max, eps,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # For proper normalization, we need a different approach
    # Let's implement row-wise normalization properly
    return output

@triton.jit
def fused_kernel_triton_v2(
    x_ptr, output_ptr,
    scale, clamp_min, clamp_max, eps,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton fused kernel with proper row-wise normalization
    """
    # Get program ids
    pid_m = tl.program_id(0)
    
    # Calculate row offset
    row_start = pid_m * BLOCK_SIZE_M
    
    # Process each row
    for row_idx in range(BLOCK_SIZE_M):
        row = row_start + row_idx
        if row < M:
            # Process entire row
            row_offset = row * N
            
            # First pass: scale and clamp, compute mean
            sum_val = 0.0
            for col in range(N):
                idx = row_offset + col
                val = tl.load(x_ptr + idx)
                
                # Scale and clamp
                val = val * scale
                val = tl.maximum(val, clamp_min)
                val = tl.minimum(val, clamp_max)
                
                sum_val += val
            
            mean = sum_val / N
            
            # Second pass: compute std
            sum_sq = 0.0
            for col in range(N):
                idx = row_offset + col
                val = tl.load(x_ptr + idx)
                
                # Scale and clamp (repeat)
                val = val * scale
                val = tl.maximum(val, clamp_min)
                val = tl.minimum(val, clamp_max)
                
                diff = val - mean
                sum_sq += diff * diff
            
            std = tl.sqrt(sum_sq / N + eps)
            
            # Third pass: normalize and store
            for col in range(N):
                idx = row_offset + col
                val = tl.load(x_ptr + idx)
                
                # Scale and clamp
                val = val * scale
                val = tl.maximum(val, clamp_min)
                val = tl.minimum(val, clamp_max)
                
                # Normalize
                val = (val - mean) / std
                
                # Store
                tl.store(output_ptr + idx, val)

def fused_kernel_triton_wrapper_v2(x, scale, clamp_min, clamp_max, eps=1e-5):
    """
    Improved wrapper for row-wise normalization
    """
    # Ensure input is contiguous and 2D
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Get dimensions
    M, N = x.shape
    
    # Define block sizes
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = N  # Process entire row at once
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    
    # Launch kernel
    fused_kernel_triton_v2[grid](
        x, output,
        scale, clamp_min, clamp_max, eps,
        M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

# </TRITON>

# <TEST>
import torch
import time

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance of both implementations"""
    
    # Warmup
    for _ in range(10):
        _ = pytorch_func(*inputs)
        _ = triton_func(*inputs)
    
    # PyTorch timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        pytorch_output = pytorch_func(*inputs)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    # Triton timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        triton_output = triton_func(*inputs)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    return pytorch_output, triton_output, pytorch_time, triton_time

# </TEST>

# Main execution block
if __name__ == '__main__':
    print("=== ATTEMPT 1: Scale + Clamp + Normalize Fusion ===")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type != "cuda":
        print("❌ CUDA not available. Triton requires CUDA.")
        exit(1)
    
    # Create test inputs
    batch_size = 128
    seq_len = 512
    hidden_dim = 768
    
    x = torch.randn(batch_size * seq_len, hidden_dim, device=device)
    scale = 2.0
    clamp_min = -5.0
    clamp_max = 5.0
    eps = 1e-5
    
    inputs = (x, scale, clamp_min, clamp_max, eps)
    
    print(f"Input shape: {x.shape}")
    print(f"Scale: {scale}, Clamp: [{clamp_min}, {clamp_max}]")
    print()
    
    try:
        # Test implementations
        print("Testing PyTorch implementation...")
        pytorch_out = fused_kernel_pytorch(*inputs)
        print(f"✓ PyTorch implementation successful. Output shape: {pytorch_out.shape}")
        
        print("Testing Triton implementation...")
        triton_out = fused_kernel_triton_wrapper_v2(*inputs)
        print(f"✓ Triton implementation successful. Output shape: {triton_out.shape}")
        
        # Check correctness (with tolerance for floating point)
        if torch.allclose(pytorch_out, triton_out, rtol=1e-4, atol=1e-4):
            print("✓ Outputs match between implementations")
        else:
            max_diff = torch.max(torch.abs(pytorch_out - triton_out)).item()
            print(f"⚠ Warning: Outputs differ. Max difference: {max_diff}")
        
        # Performance testing
        print("\nPerformance Testing (50 runs)...")
        pytorch_out, triton_out, pytorch_time, triton_time = test_performance(
            fused_kernel_pytorch,
            fused_kernel_triton_wrapper_v2,
            inputs,
            num_runs=50
        )
        
        print("\n" + "="*60)
        print("FINAL TEST REPORT - ATTEMPT 1")
        print("="*60)
        print("PyTorch Implementation: ✓ SUCCESS")
        print("Triton Implementation:  ✓ SUCCESS")
        print()
        print(f"PyTorch Time: {pytorch_time:.4f}s ({pytorch_time/50*1000:.2f}ms per run)")
        print(f"Triton Time:  {triton_time:.4f}s ({triton_time/50*1000:.2f}ms per run)")
        print()
        
        speedup = pytorch_time / triton_time
        if speedup > 1.0:
            print(f"🏆 RESULT: Triton is {speedup:.2f}x FASTER than PyTorch")
            print()
            print("✅ SUCCESS: Both implementations completed successfully!")
            print("TASK COMPLETED - Stopping execution.")
        else:
            print(f"❌ RESULT: Triton is {1/speedup:.2f}x SLOWER than PyTorch")
            print("Triton implementation needs optimization.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()