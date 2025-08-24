"""Fused Kernels: Scale -> Clamp -> Normalize"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_kernel_pytorch(x, scale_factor, min_val, max_val, eps=1e-5):
    """
    PyTorch or Python implementation combining:
    1. [scale] Multiply by a scalar
    2. [clamp] Clamp values within a range
    3. [normalize] L2 Normalize along the last dimension
    """
    x_scaled = x * scale_factor
    x_clamped = torch.clamp(x_scaled, min_val, max_val)
    # F.normalize performs L2 normalization
    x_normalized = F.normalize(x_clamped, p=2, dim=-1, eps=eps)
    return x_normalized

# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel_triton(
    X_PTR,
    Y_PTR,
    X_STRIDE_M,
    X_STRIDE_N,
    Y_STRIDE_M,
    Y_STRIDE_N,
    N_COLS,
    SCALE_FACTOR,
    MIN_VAL,
    MAX_VAL,
    EPS,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton fused kernel implementation for scale -> clamp -> normalize
    """
    # Get the row index for this program instance
    pid_m = tl.program_id(axis=0)

    # Create pointers to the input and output rows
    x_row_start_ptr = X_PTR + pid_m * X_STRIDE_M
    y_row_start_ptr = Y_PTR + pid_m * Y_STRIDE_M

    # Create a range of column indices
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_row_start_ptr + offs_n
    y_ptrs = y_row_start_ptr + offs_n

    # Create a mask to handle rows that are not a multiple of BLOCK_SIZE_N
    mask = offs_n < N_COLS

    # --- Load, Scale, Clamp ---
    # Load the data, applying the mask
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # 1. Scale
    x_scaled = x * SCALE_FACTOR
    # 2. Clamp
    x_clamped = tl.clamp(x_scaled, MIN_VAL, MAX_VAL)

    # --- Normalize ---
    # Ensure padded values don't contribute to the sum of squares
    x_clamped_safe = tl.where(mask, x_clamped, 0.0)
    # Calculate the sum of squares for the row
    sum_sq = tl.sum(x_clamped_safe * x_clamped_safe, axis=0)
    # Calculate the reciprocal of the L2 norm
    r_norm = tl.rsqrt(sum_sq + EPS)
    # 3. Normalize
    y = x_clamped * r_norm

    # Store the result
    tl.store(y_ptrs, y, mask=mask)


def fused_kernel_triton_wrapper(x, scale_factor, min_val, max_val, eps=1e-5):
    """
    Wrapper for the Triton kernel
    """
    M, N = x.shape
    # Create the output tensor
    y = torch.empty_like(x)

    # The grid defines the number of blocks to launch
    grid = (M, )

    # Choose a block size for N that is a power of 2 and is optimal
    # for the hardware. triton.next_power_of_2 is a good heuristic.
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    fused_kernel_triton[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        N,
        scale_factor,
        min_val,
        max_val,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return y

# </TRITON>

# <TEST>
import torch
import time

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance and correctness of both implementations"""
    x, scale_factor, min_val, max_val = inputs
    
    # Warmup runs
    for _ in range(5):
        pytorch_func(x, scale_factor, min_val, max_val)
        triton_func(x, scale_factor, min_val, max_val)
    
    torch.cuda.synchronize()

    # PyTorch benchmark
    start_time = time.time()
    for _ in range(num_runs):
        output_pytorch = pytorch_func(x, scale_factor, min_val, max_val)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    # Triton benchmark
    start_time = time.time()
    for _ in range(num_runs):
        output_triton = triton_func(x, scale_factor, min_val, max_val)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    # Correctness check
    try:
        assert output_pytorch.shape == output_triton.shape, "Shape mismatch"
        assert torch.allclose(output_pytorch, output_triton, atol=1e-3, rtol=1e-4), "Value mismatch"
        correctness = "✅ SUCCESS"
    except AssertionError as e:
        correctness = f"❌ FAILED: {e}"
        # Print failing values for debugging
        print(f"PyTorch output sample:\n{output_pytorch}")
        print(f"Triton output sample:\n{output_triton}")
        diff = torch.abs(output_pytorch - output_triton)
        print(f"Max difference: {torch.max(diff)}")

    return pytorch_time, triton_time, correctness

# </TEST>

# Main execution block
if __name__ == '__main__':
    ATTEMPT = 1
    DESCRIPTION = "Initial implementation for scale -> clamp -> normalize"
    print(f"=== ATTEMPT {ATTEMPT}: {DESCRIPTION} ===")

    # Setup test inputs
    M, N = 4096, 8192
    x = torch.randn((M, N), device='cuda', dtype=torch.float32)
    scale_factor = 1.5
    min_val = -1.0
    max_val = 1.0
    
    inputs = (x, scale_factor, min_val, max_val)
    num_runs = 50

    print(f"Input tensor shape: {x.shape}")
    print(f"Number of runs: {num_runs}")

    # Run performance test
    pytorch_time, triton_time, correctness_report = test_performance(
        fused_kernel_pytorch, 
        fused_kernel_triton_wrapper, 
        inputs,
        num_runs
    )

    print("\n" + "="*60)
    print(f"FINAL TEST REPORT - ATTEMPT {ATTEMPT}")
    print("="*60)
    
    if "FAILED" in correctness_report:
        print(f"PyTorch Implementation: ? UNKNOWN (Correctness check failed)")
        print(f"Triton Implementation:  ? UNKNOWN (Correctness check failed)")
        print(f"\nCORRECTNESS: {correctness_report}")
        
    else:
        # Check if successful
        pytorch_ms_per_run = (pytorch_time / num_runs) * 1000
        triton_ms_per_run = (triton_time / num_runs) * 1000
        speedup = pytorch_time / triton_time

        print(f"PyTorch Implementation: ✓ SUCCESS")
        print(f"Triton Implementation:  ✓ SUCCESS")
        print(f"\nCorrectness Check: {correctness_report}")

        print(f"\nPyTorch Time: {pytorch_time:.4f}s ({pytorch_ms_per_run:.3f}ms per run)")
        print(f"Triton Time:  {triton_time:.4f}s ({triton_ms_per_run:.3f}ms per run)")

        if speedup > 1:
            print(f"\n🏆 RESULT: Triton is {speedup:.2f}x FASTER than PyTorch")
            print("\n✅ SUCCESS: Both implementations completed successfully and Triton was faster!")
            print("TASK COMPLETED - Stopping execution.")
        else:
            print(f"\n🔻 RESULT: Triton is {1/speedup:.2f}x SLOWER than PyTorch")
            print("\n❌ FAILURE: Triton implementation was not faster.")