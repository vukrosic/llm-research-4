"""Fused MLP block: tanh(x @ W1 + b1) @ W2 + b2"""

# <PYTHON>
def mlp_block(x, w1, b1, w2, b2):
    # Five separate kernels in PyTorch:
    hidden = torch.matmul(x, w1)      # Kernel 1: matmul
    hidden = hidden + b1              # Kernel 2: add bias
    hidden = torch.tanh(hidden)       # Kernel 3: tanh activation
    output = torch.matmul(hidden, w2) # Kernel 4: matmul
    output = output + b2              # Kernel 5: add bias
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def mlp_block_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, output_ptr,
    M, N, K, L,
    stride_xm, stride_xk,
    stride_w1k, stride_w1l,
    stride_w2l, stride_w2n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr
):
    # Single fused kernel doing the entire MLP computation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # First linear layer: x @ W1 + b1
    acc1 = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # Load blocks
        a = tl.load(x_ptr + ...)
        b = tl.load(w1_ptr + ...)
        acc1 += tl.dot(a, b)
    
    # Add bias and apply tanh activation
    bias1 = tl.load(b1_ptr + tl.arange(0, BLOCK_L))
    hidden = acc1 + bias1[None, :]
    hidden = tl.tanh(hidden)
    
    # Second linear layer: hidden @ W2 + b2
    # (In practice, you would store the hidden values to a temporary buffer and load from there)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for l in range(0, L, BLOCK_L):
        # Compute offset for the current block
        hidden_offset = pid_m * BLOCK_M * L + l * BLOCK_M
        w2_offset = l * stride_w2l + pid_n * BLOCK_N * stride_w2n
        
        # Load blocks
        a = tl.load(output_ptr + hidden_offset + tl.arange(0, BLOCK_M), mask=tl.arange(0, BLOCK_M) < BLOCK_M)
        b = tl.load(w2_ptr + w2_offset + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < BLOCK_N)
        acc2 += tl.dot(a, b)
    
    # Add bias
    bias2 = tl.load(b2_ptr + tl.arange(0, BLOCK_N))
    output = acc2 + bias2[None, :]
    
    # Store result
    tl.store(output_ptr + ..., output)

def mlp_block(x, w1, b1, w2, b2):
    # Wrapper that launches the fused kernel
    M, K = x.shape
    K, L = w1.shape
    L, N = w2.shape
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    mlp_block_kernel[grid](
        x, w1, b1, w2, b2, output,
        M, N, K, L,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, BLOCK_L=32
    )
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 128
    in_features = 768
    hidden_features = 3072
    out_features = 768
    
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    w1 = torch.randn(in_features, hidden_features, device='cuda', dtype=torch.float16)
    b1 = torch.randn(hidden_features, device='cuda', dtype=torch.float16)
    w2 = torch.randn(hidden_features, out_features, device='cuda', dtype=torch.float16)
    b2 = torch.randn(out_features, device='cuda', dtype=torch.float16)
    
    return (x, w1, b1, w2, b2)
# </TEST>
