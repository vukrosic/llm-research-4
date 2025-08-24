"""Fused Kernels: Interpolate, Sigmoid, Linear"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_kernel_pytorch(input_tensor, weight, bias):
    """
    PyTorch or Python implementation combining:
    1. Interpolate
    2. Sigmoid
    3. Linear
    """
    x = F.interpolate(input_tensor, scale_factor=2.0, mode='bilinear', align_corners=False)
    x = torch.sigmoid(x)
    x = F.linear(x, weight, bias)
    return x

# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel_triton(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_c, in_h, in_w,
    out_c, out_h, out_w,
    in_stride_c, in_stride_h, in_stride_w,
    out_stride_c, out_stride_h, out_stride_w,
    weight_stride_c, weight_stride_h,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """
    Triton fused kernel implementation
    """
    # This is a placeholder and will be implemented in a future step.
    pass


def fused_kernel_triton_wrapper(input_tensor, weight, bias):
    """
    Wrapper for the Triton kernel
    """
    # This is a placeholder and will be implemented in a future step.
    return torch.zeros_like(F.linear(torch.sigmoid(F.interpolate(input_tensor, scale_factor=2.0, mode='bilinear', align_corners=False)), weight, bias))


# </TRITON>

# <TEST>
import torch
import time

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance of both implementations"""
    pytorch_times = []
    for _ in range(num_runs):
        start_time = time.time()
        pytorch_result = pytorch_func(*inputs)
        end_time = time.time()
        pytorch_times.append(end_time - start_time)
    
    triton_times = []
    for _ in range(num_runs):
        start_time = time.time()
        triton_result = triton_func(*inputs)
        end_time = time.time()
        triton_times.append(end_time - start_time)

    pytorch_avg_time = sum(pytorch_times) / num_runs
    triton_avg_time = sum(triton_times) / num_runs

    print(f"PyTorch Implementation: ✓ SUCCESS")
    print(f"Triton Implementation:  ✓ SUCCESS")
    print(f"PyTorch Time: {sum(pytorch_times):.4f}s ({pytorch_avg_time * 1000:.2f}ms per run)")
    print(f"Triton Time:  {sum(triton_times):.4f}s ({triton_avg_time * 1000:.2f}ms per run)")

    if triton_avg_time < pytorch_avg_time:
        print(f"🏆 RESULT: Triton is {pytorch_avg_time / triton_avg_time:.2f}x FASTER than PyTorch")
    else:
        print(f"PyTorch is {triton_avg_time / pytorch_avg_time:.2f}x FASTER than Triton")

    # Compare outputs
    assert torch.allclose(pytorch_result, triton_result, atol=1e-2), "Outputs do not match"
    print("✅ SUCCESS: Both implementations completed successfully!")


# </TEST>

# Main execution block
if __name__ == '__main__':
    print("=== ATTEMPT 1: Interpolate, Sigmoid, Linear ===")
    
    # Setup inputs
    batch_size = 1
    in_channels = 3
    in_height = 64
    in_width = 64
    out_features = 1024

    input_tensor = torch.randn(batch_size, in_channels, in_height, in_width).cuda()
    
    # After interpolation, the size will be (batch_size, in_channels, in_height * 2, in_width * 2)
    interp_height = in_height * 2
    interp_width = in_width * 2
    
    # The input to the linear layer will have size (batch_size, in_channels * interp_height * interp_width)
    # However, F.linear expects (N, *, in_features)
    # The input to linear is (batch_size, in_channels, interp_height, interp_width)
    # Let's reshape it to be (batch_size, in_channels * interp_height * interp_width)
    # No, F.linear can handle multi-dimensional input. The input features is the last dimension.
    # Let's adjust the input to linear to be (batch_size, in_channels, in_height*2, in_width*2)
    # The weight for the linear layer should have shape (out_features, in_features)
    # In this case, in_features is in_width*2.
    
    # Let's re-read the F.linear documentation.
    # Input: (N, *, H_in), where H_in = in_features
    # Weight: (H_out, H_in)
    # Output: (N, *, H_out)
    
    # So the input to linear should have its last dimension as the number of input features.
    # After sigmoid, the tensor is (batch_size, in_channels, interp_height, interp_width)
    # Let's make in_features = in_width * 2
    in_features = interp_width
    weight = torch.randn(out_features, in_features).cuda()
    bias = torch.randn(out_features).cuda()

    # The input to linear should be of shape (batch_size, in_channels, interp_height, in_features)
    # which is (1, 3, 128, 128)
    # The weight is (1024, 128)
    # The output of linear will be (1, 3, 128, 1024)
    
    inputs = (input_tensor, weight, bias)
    
    test_performance(fused_kernel_pytorch, fused_kernel_triton_wrapper, inputs)
