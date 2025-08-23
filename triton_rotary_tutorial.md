# Triton Rotary Embeddings Implementation Tutorial

This tutorial explains the Triton implementation of Rotary Positional Embeddings (RoPE) in the context of transformer attention mechanisms. We'll break down the implementation step by step, explaining each part in detail.

## What are Rotary Embeddings?

Rotary Positional Embeddings (RoPE) are a method for incorporating positional information into transformer models. Unlike traditional positional embeddings that are added to the input, RoPE applies rotations to the query and key vectors in the attention mechanism.

The core idea is to rotate pairs of elements in the embedding vectors based on their position. For a 2D vector [x, y], the rotation is:
- x' = x * cos(θ) - y * sin(θ)
- y' = x * sin(θ) + y * cos(θ)

Where θ depends on the position and the dimension of the embedding.

## Triton Implementation Overview

The Triton implementation consists of two main parts:
1. The `rotary_kernel` - a low-level CUDA kernel written in Triton language
2. The `TritonRotary` class - a PyTorch module that uses the kernel

Let's examine each part in detail.

## 1. The Rotary Kernel

```python
@triton.jit
def rotary_kernel(
    Q, K,
    cos, sin,
    seqlen,
    d_head,
    BLOCK_SIZE: tl.constexpr,
):
```

This is the core Triton kernel that applies rotary embeddings to query and key tensors.

### Parameters Explanation:

- `Q`: The query tensor in the attention mechanism
- `K`: The key tensor in the attention mechanism
- `cos`: Precomputed cosine values for all positions and dimensions
- `sin`: Precomputed sine values for all positions and dimensions
- `seqlen`: Length of the sequence (number of tokens)
- `d_head`: Dimension of each attention head
- `BLOCK_SIZE`: Compile-time constant that determines the block size for processing

### Kernel Logic:

```python
# Calculate program indices
batch_head_idx = tl.program_id(0)
seq_idx = tl.program_id(1)
```

In Triton, each kernel instance is identified by program IDs:
- `batch_head_idx`: Identifies which batch and head we're processing
- `seq_idx`: Identifies which position in the sequence we're processing

```python
# Calculate memory offsets
q_offset = batch_head_idx * seqlen * d_head + seq_idx * d_head
k_offset = batch_head_idx * seqlen * d_head + seq_idx * d_head
```

These offsets determine where in memory to find the query and key values for this specific kernel instance.

```python
# Process in chunks
for i in range(0, d_head // 2, BLOCK_SIZE):
    idx = i + tl.arange(0, BLOCK_SIZE)
    mask = idx < d_head // 2
```

We process the embedding dimensions in chunks of `BLOCK_SIZE`. Since rotary embeddings work on pairs of elements, we only need to process `d_head // 2` pairs.

```python
# Load Q and K values
q1 = tl.load(Q + q_offset + idx, mask=mask, other=0.0)
q2 = tl.load(Q + q_offset + idx + d_head // 2, mask=mask, other=0.0)
k1 = tl.load(K + k_offset + idx, mask=mask, other=0.0)
k2 = tl.load(K + k_offset + idx + d_head // 2, mask=mask, other=0.0)
```

We load pairs of elements from both query and key tensors:
- `q1`, `k1`: First elements of each pair
- `q2`, `k2`: Second elements of each pair

```python
# Load cos and sin
c = tl.load(cos + seq_idx * (d_head // 2) + idx, mask=mask, other=0.0)
s = tl.load(sin + seq_idx * (d_head // 2) + idx, mask=mask, other=0.0)
```

We load the precomputed cosine and sine values for the current sequence position and dimensions.

```python
# Apply rotation
q1_new = q1 * c + q2 * s
q2_new = q1 * (-s) + q2 * c
k1_new = k1 * c + k2 * s
k2_new = k1 * (-s) + k2 * c
```

This is the core rotary embedding calculation, applying the rotation formulas to both query and key pairs.

```python
# Store results
tl.store(Q + q_offset + idx, q1_new, mask=mask)
tl.store(Q + q_offset + idx + d_head // 2, q2_new, mask=mask)
tl.store(K + k_offset + idx, k1_new, mask=mask)
tl.store(K + k_offset + idx + d_head // 2, k2_new, mask=mask)
```

Finally, we store the rotated values back to memory.

## 2. The TritonRotary Class

```python
class TritonRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        
        # Precompute cos and sin
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
```

### Initialization:

- `dim`: The dimension of the attention head
- `max_seq_len`: Maximum sequence length we'll process

The initialization precomputes all the cosine and sine values we'll need:
- `angular_freq`: Frequencies for different dimensions, following the RoPE paper formula
- `t`: Position indices from 0 to max_seq_len-1
- `theta`: Matrix of angles for each position and dimension
- `cos`, `sin`: Precomputed cosine and sine values stored as buffers

### Forward Pass:

```python
def forward(self, q, k):
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Reshape for kernel
    q = q.reshape(batch_size * n_heads, seq_len, d_head)
    k = k.reshape(batch_size * n_heads, seq_len, d_head)
```

We reshape the tensors to match the kernel's expected format:
- `batch_size`: Number of sequences in the batch
- `n_heads`: Number of attention heads
- `seq_len`: Length of each sequence
- `d_head`: Dimension of each attention head

```python
# Apply rotary embeddings
grid = (batch_size * n_heads, seq_len)
BLOCK_SIZE = triton.next_power_of_2(d_head // 2)

rotary_kernel[grid](
    q, k,
    self.cos, self.sin,
    seq_len, d_head,
    BLOCK_SIZE=BLOCK_SIZE,
)
```

We launch the Triton kernel with:
- `grid`: Specifies how many kernel instances to launch (one per batch-head and sequence position)
- `BLOCK_SIZE`: Uses the next power of 2 for efficient GPU processing

```python
# Reshape back
q = q.reshape(batch_size, n_heads, seq_len, d_head)
k = k.reshape(batch_size, n_heads, seq_len, d_head)

return q, k
```

Finally, we reshape the tensors back to their original format and return them.

## Key Variables Explained

Let's break down the important variables:

### In the kernel:
- `Q`, `K`: Input tensors containing query and key vectors
- `cos`, `sin`: Precomputed rotation values
- `seqlen`: How many tokens are in each sequence
- `d_head`: Size of each attention head vector
- `BLOCK_SIZE`: Processing chunk size for GPU efficiency

### In the class:
- `dim`: Dimension of attention head vectors
- `max_seq_len`: Maximum sequence length we support
- `angular_freq`: Different frequencies for different dimensions
- `t`: Position indices (0, 1, 2, ..., max_seq_len-1)
- `theta`: Matrix of angles for rotation calculations

## Why Use Triton?

Triton provides several advantages over pure PyTorch implementations:

1. **Performance**: Triton kernels can be significantly faster than PyTorch operations
2. **Memory Efficiency**: Direct memory access without intermediate tensors
3. **Parallelization**: Each kernel instance processes one position independently
4. **GPU Optimization**: Triton is designed specifically for GPU execution

## How It Works in Practice

When using rotary embeddings in a transformer:

1. During model initialization, we create a `TritonRotary` module
2. In the attention forward pass, we pass query and key tensors to this module
3. The module reshapes tensors for efficient processing
4. The Triton kernel applies rotations in parallel across all positions
5. The rotated tensors are returned for use in attention computation

This implementation efficiently handles the mathematical transformations required for rotary embeddings while leveraging GPU parallelization through Triton.
