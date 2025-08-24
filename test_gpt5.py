"""Fused Kernels: scale -> clamp -> normalize (per-row)"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_kernel_pytorch(x, scale, clamp_min, clamp_max, eps=1e-5):
    """
    PyTorch or Python implementation combining:
    1. scale (elementwise multiply by `scale`)
    2. clamp (clip to [clamp_min, clamp_max])
    3. normalize (per-row: (x - mean) / sqrt(var + eps))
    """
    x = x * scale
    x = torch.clamp(x, clamp_min, clamp_max)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, unbiased=False, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return x

# </PYTHON>

# <TRITON>
import torch
try:
    import triton
    import triton.language as tl
except Exception as e:
    triton = None
    tl = None
    _triton_import_error = e

if triton is not None:
    @triton.jit
    def fused_kernel_triton_kernel(
        X_ptr,  # *f32
        Y_ptr,  # *f32
        scale,  # f32
        clamp_min,  # f32
        clamp_max,  # f32
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offs = row_id * n_cols + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < n_cols

        # Load a row chunk (assuming n_cols <= BLOCK_SIZE for simplicity in this first attempt)
        x = tl.load(X_ptr + offs, mask=mask, other=0.0)

        # 1) scale
        x = x * scale
        # 2) clamp
        x = tl.maximum(x, clamp_min)
        x = tl.minimum(x, clamp_max)

        # 3) per-row normalize
        # compute mean and var over the row (again, assuming n_cols <= BLOCK_SIZE)
        # reduce
        mean = tl.sum(x, axis=0) / n_cols
        diff = x - mean
        var = tl.sum(diff * diff, axis=0) / n_cols
        x = diff / tl.sqrt(var + eps)

        tl.store(Y_ptr + offs, x, mask=mask)

def fused_kernel_triton_wrapper(x, scale, clamp_min, clamp_max, eps=1e-5):
    """
    Wrapper for the Triton kernel
    """
    if triton is None:
        raise ImportError(f"Triton is not available: {_triton_import_error}")
    assert x.is_cuda, "Input must be on CUDA for Triton kernel"
    B, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = 1 << (N - 1).bit_length()  # next power of two >= N
    grid = (B,)
    fused_kernel_triton_kernel[grid](
        x, y, scale, clamp_min, clamp_max,
        n_cols=N,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps
    )
    return y

# </TRITON>

# <TEST>
import time
import traceback

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance of both implementations"""
    x, scale, clamp_min, clamp_max, eps = inputs

    # Warmup for PyTorch
    for _ in range(10):
        y_ref = pytorch_func(x, scale, clamp_min, clamp_max, eps)

    # Time PyTorch
    t0 = time.time()
    for _ in range(num_runs):
        y_ref = pytorch_func(x, scale, clamp_min, clamp_max, eps)
    t1 = time.time()
    pytorch_time = t1 - t0

    # Compare Triton (if available)
    triton_time = None
    y_triton = None
    triton_error = None
    try:
        # if CUDA available, move to CUDA for triton
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            torch.cuda.synchronize()
            for _ in range(10):
                y_triton = triton_func(x_cuda, scale, clamp_min, clamp_max, eps)
            torch.cuda.synchronize()
            t2 = time.time()
            for _ in range(num_runs):
                y_triton = triton_func(x_cuda, scale, clamp_min, clamp_max, eps)
            torch.cuda.synchronize()
            t3 = time.time()
            triton_time = t3 - t2
            # move back to cpu for comparison
            y_triton_cpu = y_triton.cpu()
        else:
            raise RuntimeError("CUDA is not available for Triton benchmark.")
    except Exception as e:
        triton_error = traceback.format_exc()
        y_triton_cpu = None

    # Correctness check (if we have Triton output)
    max_abs_diff = None
    if triton_time is not None and y_triton_cpu is not None:
        y_ref_now = pytorch_func(x, scale, clamp_min, clamp_max, eps)
        max_abs_diff = (y_ref_now - y_triton_cpu).abs().max().item()

    return {
        "pytorch_time": pytorch_time,
        "triton_time": triton_time,
        "max_abs_diff": max_abs_diff,
        "triton_error": triton_error,
    }

# </TEST>

# Main execution block
if __name__ == '__main__':
    ATTEMPT = 1
    DESCRIPTION = "scale -> clamp -> normalize (per-row)"
    print(f"=== ATTEMPT {ATTEMPT}: {DESCRIPTION} ===")

    # Generate test inputs
    torch.manual_seed(0)
    B, N = 1024, 256  # batch of rows, row length
    x = torch.randn(B, N, dtype=torch.float32)
    scale = 1.2345
    clamp_min = -0.5
    clamp_max = 0.5
    eps = 1e-5

    # Run tests
    results = test_performance(
        fused_kernel_pytorch,
        fused_kernel_triton_wrapper,
        (x, scale, clamp_min, clamp_max, eps),
        num_runs=30,
    )

    print("PyTorch Implementation: SUCCESS")
    print(f"PyTorch Time: {results['pytorch_time']:.4f}s ({results['pytorch_time']*1000/30:.2f}ms per run)")
    if results['triton_time'] is None:
        print("Triton Implementation: FAILURE")
        print("Reason:\n", results['triton_error'])
    else:
        print("Triton Implementation: SUCCESS")
        print(f"Triton Time: {results['triton_time']:.4f}s ({results['triton_time']*1000/30:.2f}ms per run)")
        print(f"Max abs diff vs PyTorch: {results['max_abs_diff']:.3e}")
        speedup = results['pytorch_time'] / results['triton_time']
        print(f"RESULT: Triton is {speedup:.2f}x faster than PyTorch")