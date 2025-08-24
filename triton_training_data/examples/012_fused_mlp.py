"""Fused MLP block: linear -> activation -> linear"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_mlp(x, weight1, bias1, weight2, bias2, activation='gelu'):
    # Multiple separate kernels in PyTorch:
    x = torch.matmul(x, weight1)       # Kernel 1: matmul
    x = x + bias1                      # Kernel 2: add bias
    if activation == 'gelu':
        x = F.gelu(x)                  # Kernel 3: gelu
    elif activation == 'relu':
        x = F.relu(x)                  # Kernel 3: relu
    x = torch.matmul(x, weight2)       # Kernel 4: matmul
    x = x + bias2                      # Kernel 5: add bias
    return x
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def mlp_kernel(
    x_ptr, weight1_ptr, bias1_ptr, weight2_ptr, bias2_ptr, output_ptr,
    M, N, K, P,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w2n, stride_w2p,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_P: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    
    # First linear layer: x @ weight1 + bias1
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load x and weight1
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w1_ptrs = weight1_ptr + (offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
    
    for k in range(0, K, BLOCK_K):
        x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w1_block = tl.load(w1_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc1 += tl.dot(x_block, w1_block)
        x_ptrs += BLOCK_K * stride_xk
        w1_ptrs += BLOCK_K * stride_w1k
    
    # Add bias1
    bias1_ptrs = bias1_ptr + offs_n
    bias1 = tl.load(bias1_ptrs, mask=offs_n < N, other=0.0)
    hidden = acc1 + bias1[None, :]
    
    # Apply activation
    if ACTIVATION == "gelu":
        hidden = tl.sigmoid(1.702 * hidden.to(tl.float32)) * hidden  # Approximate GELU
    elif ACTIVATION == "relu":
        hidden = tl.maximum(hidden, 0.0)
    
    # Second linear layer: hidden @ weight2 + bias2
    acc2 = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
    
    # Load weight2
    w2_ptrs = weight2_ptr + (offs_n[:, None] * stride_w2n + offs_p[None, :] * stride_w2p)
    
    for n in range(0, N, BLOCK_N):
        # Calculate the correct offset for the hidden tensor
        hidden_offset = n  # Start of the current block in the N dimension
        hidden_ptrs = hidden + (offs_m[:, None] * N + (hidden_offset + offs_n[None, :]))
        hidden_block = tl.load(hidden_ptrs, 
                              mask=(offs_m[:, None] < M) & ((hidden_offset + offs_n[None, :]) < N), other=0.0)
        w2_block = tl.load(w2_ptrs, mask=((n + offs_n[:, None]) < N) & (offs_p[None, :] < P), other=0.0)
        acc2 += tl.dot(hidden_block, w2_block)
        w2_ptrs += BLOCK_N * stride_w2n
    
    # Add bias2
    bias2_ptrs = bias2_ptr + offs_p
    bias2 = tl.load(bias2_ptrs, mask=offs_p < P, other=0.0)
    output = acc2 + bias2[None, :]
    
    # Store result
    output_ptrs = output_ptr + (offs_m[:, None] * P + offs_p[None, :])
    tl.store(output_ptrs, output.to(x_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_p[None, :] < P))

def fused_mlp(x, weight1, bias1, weight2, bias2, activation='gelu'):
    M, K = x.shape
    K, N = weight1.shape
    N, P = weight2.shape
    output = torch.empty((M, P), device=x.device, dtype=x.dtype)
    
    grid = (triton.cdiv(M, 32), triton.cdiv(P, 32))
    
    mlp_kernel[grid](
        x, weight1, bias1, weight2, bias2, output,
        M, N, K, P,
        x.stride(0), x.stride(1),
        weight1.stride(0), weight1.stride(1),
        weight2.stride(0), weight2.stride(1),
        ACTIVATION=activation,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, BLOCK_P=32
    )
    return output
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 32
    in_features = 768
    hidden_features = 3072
    out_features = 768
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    weight1 = torch.randn(in_features, hidden_features, device='cuda', dtype=torch.float16)
    bias1 = torch.randn(hidden_features, device='cuda', dtype=torch.float16)
    weight2 = torch.randn(hidden_features, out_features, device='cuda', dtype=torch.float16)
    bias2 = torch.randn(out_features, device='cuda', dtype=torch.float16)
    return (x, weight1, bias1, weight2, bias2)
# </TEST>