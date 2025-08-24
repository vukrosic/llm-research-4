"""Fused Kernels: Flatten, Matmul, Swish"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_kernel_pytorch(input_tensor, weight_matrix):
    """
    PyTorch or Python implementation combining:
    1. Flatten
    2. Matmul
    3. Swish
    """
    flattened_tensor = torch.flatten(input_tensor, start_dim=1)
    matmul_result = torch.matmul(flattened_tensor, weight_matrix)
    output = F.silu(matmul_result)
    return output

# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel_triton(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    input_features,
    output_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton fused kernel implementation
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(output_features, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % batch_size
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % output_features
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_am[:, None] * input_features + offs_k[None, :])
    weight_ptrs = weight_ptr + (offs_k[:, None] * output_features + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(input_features, BLOCK_SIZE_K)):
        a = tl.load(input_ptrs, mask=(offs_k[None, :] < input_features), other=0.0)
        b = tl.load(weight_ptrs, mask=(offs_k[:, None] < input_features), other=0.0)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K
        weight_ptrs += BLOCK_SIZE_K * output_features

    # Swish activation
    result = accumulator * tl.sigmoid(accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + output_features * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < batch_size) & (offs_cn[None, :] < output_features)
    tl.store(output_ptrs, result, mask=c_mask)


def fused_kernel_triton_wrapper(input_tensor, weight_matrix):
    """
    Wrapper for the Triton kernel
    """
    batch_size = input_tensor.shape[0]
    input_features = input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
    output_features = weight_matrix.shape[1]
    
    input_tensor_flat = input_tensor.view(batch_size, -1)

    output = torch.empty(batch_size, output_features, device=input_tensor.device)
    
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_SIZE_M']) * triton.cdiv(output_features, META['BLOCK_SIZE_N']),
    )
    
    fused_kernel_triton[grid](
        input_tensor_flat,
        weight_matrix,
        output,
        batch_size,
        input_features,
        output_features,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
        GROUP_SIZE_M=8,
    )
    return output

# </TRITON>

# <TEST>
import torch
import time

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance of both implementations"""
    input_tensor, weight_matrix = inputs
    
    # Warm-up runs
    for _ in range(5):
        pytorch_func(input_tensor, weight_matrix)
        triton_func(input_tensor, weight_matrix)

    # PyTorch performance
    start_time = time.time()
    for _ in range(num_runs):
        pytorch_output = pytorch_func(input_tensor, weight_matrix)
    end_time = time.time()
    pytorch_time = end_time - start_time

    # Triton performance
    start_time = time.time()
    for _ in range(num_runs):
        triton_output = triton_func(input_tensor, weight_matrix)
    end_time = time.time()
    triton_time = end_time - start_time
    
    # Functional correctness check
    assert torch.allclose(pytorch_output, triton_output, atol=1e-2, rtol=1e-2), "Outputs do not match!"

    return pytorch_time, triton_time

# </TEST>

# Main execution block
if __name__ == '__main__':
    print("=== ATTEMPT 1: Flatten, Matmul, Swish ===")
    
    # Setup test inputs
    batch_size = 64
    input_channels = 3
    input_height = 32
    input_width = 32
    output_features = 128
    
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width).cuda()
    weight_matrix = torch.randn(input_channels * input_height * input_width, output_features).cuda()

    try:
        pytorch_time, triton_time = test_performance(
            fused_kernel_pytorch, 
            fused_kernel_triton_wrapper, 
            (input_tensor, weight_matrix)
        )

        print(f"PyTorch Implementation: ✓ SUCCESS")
        print(f"Triton Implementation:  ✓ SUCCESS")
        print(f"PyTorch Time: {pytorch_time:.4f}s ({pytorch_time/50*1000:.4f}ms per run)")
        print(f"Triton Time:  {triton_time:.4f}s ({triton_time/50*1000:.4f}ms per run)")

        if triton_time < pytorch_time:
            print(f"🏆 RESULT: Triton is {pytorch_time/triton_time:.2f}x FASTER than PyTorch")
        else:
            print(f"RESULT: PyTorch is {triton_time/pytorch_time:.2f}x FASTER than Triton")
        
        print("✅ SUCCESS: Both implementations completed successfully!")

    except Exception as e:
        print(f"❌ FAILURE: {e}")
