"""Simple Fused Kernels: Combining Basic Operations Reliably"""

# <PYTHON>
import torch
import torch.nn.functional as F

def simple_fused_kernel_pytorch(x, weight, bias, p=0.1):
    """
    Simple fused kernel combining:
    1. Linear transformation
    2. ReLU activation
    3. Dropout
    """
    # Fused operation: linear + relu + dropout
    # Note: F.linear expects weight to be (out_features, in_features)
    # but our weight is (in_features, out_features), so we need to transpose
    output = F.linear(x, weight.t(), bias)
    output = F.relu(output)
    output = F.dropout(output, p, training=True)
    return output

def multi_fused_kernel_pytorch(x, weight1, weight2, bias1, bias2, p=0.1):
    """
    Multi-fused kernel combining:
    1. First linear + relu
    2. Second linear + relu
    3. Dropout
    """
    # First fused operation
    hidden = F.linear(x, weight1.t(), bias1)
    hidden = F.relu(hidden)
    
    # Second fused operation
    output = F.linear(hidden, weight2.t(), bias2)
    output = F.relu(output)
    
    # Final dropout
    output = F.dropout(output, p, training=True)
    return output

# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def simple_fused_kernel_triton(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K, p, seed,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Simple fused kernel: linear + relu + dropout
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        
        x_chunk = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w_chunk = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        acc += tl.dot(x_chunk, w_chunk)
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n
    bias_chunk = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    output = acc + bias_chunk[None, :]
    
    # ReLU activation
    output = tl.maximum(output, 0.0)
    
    # Dropout
    random = tl.rand(seed, offs_m[:, None] * N + offs_n[None, :])
    dropout_mask = random > p
    output = tl.where(dropout_mask, output / (1 - p), 0.0)
    
    # Store result
    output_ptrs = output_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(output_ptrs, output, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def simple_fused_kernel_triton_wrapper(x, weight, bias, p=0.1):
    """
    Wrapper for the Triton simple fused kernel
    """
    M, K = x.shape
    K, N = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    simple_fused_kernel_triton[grid](
        x, weight, bias, output,
        M, N, K, p,
        torch.randint(0, 10000, (1,)).item(),
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        bias.stride(0),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    
    return output

# </TRITON>

# <TEST>
import torch
import time

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance of both implementations"""
    # Warm-up
    for _ in range(10):
        pytorch_func(*inputs)
        triton_func(*inputs)
    
    torch.cuda.synchronize()
    
    # PyTorch performance
    start_time = time.time()
    for _ in range(num_runs):
        pytorch_output = pytorch_func(*inputs)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Triton performance
    start_time = time.time()
    for _ in range(num_runs):
        triton_output = triton_func(*inputs)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    return pytorch_time, triton_time

# </TEST>

# Make the file executable by adding main execution
if __name__ == '__main__':
    print("=== ATTEMPT 1: Simple Fused Kernels ===")
    
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA is not available. Running on CPU instead.")
            device = 'cpu'
        else:
            device = 'cuda'
            print(f"Running on {device}")
        
        # Get test inputs
        if device == 'cpu':
            batch_size = 64
            in_features = 256
            out_features = 512
        else:
            batch_size = 128
            in_features = 512
            out_features = 1024
            
        x = torch.randn(batch_size, in_features, device=device, dtype=torch.float16)
        weight = torch.randn(in_features, out_features, device=device, dtype=torch.float16)
        bias = torch.randn(out_features, device=device, dtype=torch.float16)
        p = 0.1
        
        print(f"Testing Simple Fused Kernels")
        print(f"Input shape: {x.shape}")
        print(f"Weight shape: {weight.shape}")
        print(f"Hidden dimension: {in_features} -> {out_features}")
        print(f"Dropout probability: {p}")
        
        # Test basic functionality
        print("\n=== Basic Functionality Test ===")
        pytorch_success = False
        triton_success = False
        pytorch_output = None
        triton_output = None
        
        try:
            pytorch_output = simple_fused_kernel_pytorch(x, weight, bias, p)
            print(f"✓ PyTorch output shape: {pytorch_output.shape}")
            pytorch_success = True
        except Exception as e:
            print(f"✗ PyTorch failed: {e}")
        
        try:
            triton_output = simple_fused_kernel_triton_wrapper(x, weight, bias, p)
            print(f"✓ Triton output shape: {triton_output.shape}")
            triton_success = True
        except Exception as e:
            print(f"✗ Triton failed: {e}")
        
        # Performance measurement
        print("\n=== Performance Measurement ===")
        pytorch_time = None
        triton_time = None
        
        if pytorch_success:
            try:
                # Manual timing for PyTorch
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(30):
                    pytorch_output = simple_fused_kernel_pytorch(x, weight, bias, p)
                torch.cuda.synchronize()
                pytorch_time = time.time() - start_time
                print(f"✓ PyTorch timing: {pytorch_time:.4f}s ({pytorch_time/30*1000:.2f}ms per run)")
            except Exception as e:
                print(f"✗ PyTorch timing failed: {e}")
        
        if triton_success:
            try:
                # Manual timing for Triton
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(30):
                    triton_output = simple_fused_kernel_triton_wrapper(x, weight, bias, p)
                torch.cuda.synchronize()
                triton_time = time.time() - start_time
                print(f"✓ Triton timing: {triton_time:.4f}s ({triton_time/30*1000:.2f}ms per run)")
            except Exception as e:
                print(f"✗ Triton timing failed: {e}")
        
        # Final comprehensive report
        print("\n" + "="*60)
        print("FINAL TEST REPORT - ATTEMPT 1")
        print("="*60)
        
        # Implementation status
        print(f"PyTorch Implementation: {'✓ SUCCESS' if pytorch_success else '✗ FAILED'}")
        print(f"Triton Implementation:  {'✓ SUCCESS' if triton_success else '✗ FAILED'}")
        
        # Timing comparison
        if pytorch_success and triton_success:
            print(f"\nPyTorch Time: {pytorch_time:.4f}s ({pytorch_time/30*1000:.2f}ms per run)")
            print(f"Triton Time:  {triton_time:.4f}s ({triton_time/30*1000:.2f}ms per run)")
            
            if triton_time < pytorch_time:
                speedup = pytorch_time / triton_time
                print(f"\n🏆 RESULT: Triton is {speedup:.2f}x FASTER than PyTorch")
            else:
                slowdown = triton_time / pytorch_time
                print(f"\n🏆 RESULT: PyTorch is {slowdown:.2f}x FASTER than Triton")
                
        elif pytorch_success:
            print(f"\nPyTorch Time: {pytorch_time:.4f}s ({pytorch_time/30*1000:.2f}ms per run)")
            print("⚠️  Triton failed - no comparison possible")
            
        elif triton_success:
            print(f"\nTriton Time: {triton_time:.4f}s ({triton_time/30*1000:.2f}ms per run)")
            print("⚠️  PyTorch failed - no comparison possible")
            
        else:
            print("\n❌ Both implementations failed - no timing data available")
        
        print("\n=== Kernel Operations Summary ===")
        print("1. Linear transformation with bias")
        print("2. ReLU activation")
        print("3. Dropout")
        print("\nNote: Simple, reliable fused kernel implementation")
        
        if pytorch_success and triton_success:
            print("\n✅ SUCCESS: Both implementations completed successfully!")
        elif pytorch_success or triton_success:
            print(f"\n⚠️  PARTIAL SUCCESS: Only {'PyTorch' if pytorch_success else 'Triton'} completed successfully")
        else:
            print("\n❌ FAILURE: Both implementations failed")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure PyTorch and Triton are properly installed")
