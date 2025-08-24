# <PYTHON>
import torch
import torch.nn.functional as F

def fused_operation_pytorch(input1, input2, input3):
    """
    PyTorch implementation of fused operations:
    output = relu((input1 * input2) + input3)
    """
    return F.relu((input1 * input2) + input3)
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_kernel_triton(
    input1_ptr, input2_ptr, input3_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Create block pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks to handle out-of-bounds accesses
    mask = offsets < n_elements
    
    # Load data
    input1 = tl.load(input1_ptr + offsets, mask=mask)
    input2 = tl.load(input2_ptr + offsets, mask=mask)
    input3 = tl.load(input3_ptr + offsets, mask=mask)
    
    # Perform fused operations: multiply, add, then ReLU
    result = input1 * input2
    result += input3
    result = tl.maximum(result, 0.0)  # ReLU equivalent
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_operation_triton(input1, input2, input3):
    """
    Wrapper for the Triton kernel that handles:
    output = relu((input1 * input2) + input3)
    """
    # Check inputs
    assert input1.is_cuda and input2.is_cuda and input3.is_cuda, "Inputs must be on GPU"
    assert input1.shape == input2.shape == input3.shape, "Input shapes must match"
    
    # Create output tensor
    output = torch.empty_like(input1)
    
    n_elements = output.numel()
    
    # Define grid and block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _fused_kernel_triton[grid](
        input1, input2, input3, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
# </TRITON>

# <TEST>
import torch
import time
import numpy as np

def test_correctness(pytorch_func, triton_func, shape=(4096, 4096), dtype=torch.float32, tol=1e-6):
    """Test correctness of both implementations"""
    print(f"Testing correctness with shape {shape}, dtype {dtype}")
    
    # Generate random inputs
    input1 = torch.rand(shape, dtype=dtype, device='cuda')
    input2 = torch.rand(shape, dtype=dtype, device='cuda')
    input3 = torch.rand(shape, dtype=dtype, device='cuda')
    
    # Run PyTorch implementation
    with torch.no_grad():
        pytorch_output = pytorch_func(input1, input2, input3)
    
    # Run Triton implementation
    with torch.no_grad():
        triton_output = triton_func(input1, input2, input3)
    
    # Compare results
    diff = torch.abs(pytorch_output - triton_output)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < tol:
        print("✓ Correctness test PASSED")
        return True
    else:
        print("✗ Correctness test FAILED")
        return False

def test_performance(pytorch_func, triton_func, shape=(4096, 4096), dtype=torch.float32, num_runs=50, warmup=10):
    """Test performance of both implementations"""
    print(f"\nTesting performance with shape {shape}, dtype {dtype}")
    
    # Generate random inputs
    input1 = torch.rand(shape, dtype=dtype, device='cuda')
    input2 = torch.rand(shape, dtype=dtype, device='cuda')
    input3 = torch.rand(shape, dtype=dtype, device='cuda')
    
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            _ = pytorch_func(input1, input2, input3)
            _ = triton_func(input1, input2, input3)
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            pytorch_output = pytorch_func(input1, input2, input3)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms per run
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            triton_output = triton_func(input1, input2, input3)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs * 1000  # ms per run
    
    # Calculate speedup
    speedup = pytorch_time / triton_time
    
    print(f"PyTorch time: {pytorch_time:.4f} ms")
    print(f"Triton time: {triton_time:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return pytorch_time, triton_time, speedup
# </TEST>

# Main execution block
if __name__ == '__main__':
    print("Fused Kernel Implementation: Element-wise Multiplication, Addition, and ReLU")
    print("=" * 80)
    
    # Test configurations
    test_cases = [
        ((1024, 1024), torch.float32),
        ((4096, 4096), torch.float32),
        ((8192, 8192), torch.float32),
        ((1024, 1024), torch.float16),
        ((4096, 4096), torch.float16),
    ]
    
    attempt = 1
    results = []
    
    for shape, dtype in test_cases:
        print(f"\nAttempt {attempt}: Shape {shape}, dtype {dtype}")
        print("-" * 40)
        
        # Test correctness
        correctness_passed = test_correctness(
            fused_operation_pytorch, 
            fused_operation_triton, 
            shape, 
            dtype
        )
        
        # Test performance
        pytorch_time, triton_time, speedup = test_performance(
            fused_operation_pytorch, 
            fused_operation_triton, 
            shape, 
            dtype
        )
        
        # Record results
        results.append({
            'attempt': attempt,
            'shape': shape,
            'dtype': str(dtype).split('.')[-1],
            'correctness': 'PASS' if correctness_passed else 'FAIL',
            'pytorch_time_ms': pytorch_time,
            'triton_time_ms': triton_time,
            'speedup': speedup
        })
        
        attempt += 1
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS")
    print("=" * 80)
    print(f"{'Attempt':<8} {'Shape':<15} {'Dtype':<8} {'Correctness':<12} {'PyTorch (ms)':<12} {'Triton (ms)':<12} {'Speedup':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['attempt']:<8} {str(result['shape']):<15} {result['dtype']:<8} {result['correctness']:<12} "
              f"{result['pytorch_time_ms']:<12.4f} {result['triton_time_ms']:<12.4f} {result['speedup']:<8.2f}")
    
    # Calculate overall statistics
    pytorch_times = [r['pytorch_time_ms'] for r in results]
    triton_times = [r['triton_time_ms'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    print("-" * 80)
    print(f"{'Average':<8} {'':<15} {'':<8} {'':<12} {np.mean(pytorch_times):<12.4f} {np.mean(triton_times):<12.4f} {np.mean(speedups):<8.2f}")
    print(f"{'Best':<8} {'':<15} {'':<8} {'':<12} {min(pytorch_times):<12.4f} {min(triton_times):<12.4f} {max(speedups):<8.2f}")
    print(f"{'Worst':<8} {'':<15} {'':<8} {'':<12} {max(pytorch_times):<12.4f} {max(triton_times):<12.4f} {min(speedups):<8.2f}")