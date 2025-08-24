Excellent point! The real value of Triton is **kernel fusion** - combining multiple operations that would require multiple kernel launches in PyTorch into a single fused kernel. Let's restructure:

## Plan for Fused Operations Training Data

### Folder Structure
```
triton_training_data/
├── examples/
│   ├── 001_linear_relu.py         # ax + b -> relu
│   ├── 002_add_mul_add.py         # (x + y) * z + w
│   ├── 003_scaled_softmax.py      # softmax(x / sqrt(d))
│   ├── 004_swish.py               # x * sigmoid(x)
│   ├── 005_fused_gelu.py          # GELU with all ops fused
│   ├── 006_residual_layernorm.py  # layer_norm(x + residual)
│   ├── 007_attention_kernel.py    # Q@K^T/sqrt(d) + mask -> softmax
│   ├── 008_mlp_block.py           # linear -> activation -> linear
│   └── ...
├── scripts/
│   ├── validate.py
│   ├── extract.py
│   └── benchmark.py
└── output/
```

### Template for Fused Operations

**examples/001_linear_relu.py**
```python
"""Linear transformation with fused ReLU activation"""

# <PYTHON>
def linear_relu(x, weight, bias):
    # Three separate kernels in PyTorch:
    output = torch.matmul(x, weight)  # Kernel 1: matmul
    output = output + bias            # Kernel 2: add
    output = torch.relu(output)       # Kernel 3: relu
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def linear_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Single fused kernel doing matmul + bias + relu
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute one block of output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load blocks
        a = tl.load(x_ptr + ...)
        b = tl.load(weight_ptr + ...)
        acc += tl.dot(a, b)
    
    # Add bias and apply ReLU in-place
    bias = tl.load(bias_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
    output = acc + bias[None, :]
    output = tl.maximum(output, 0)  # Fused ReLU
    
    # Store result
    tl.store(output_ptr + ..., output)

def linear_relu(x, weight, bias):
    # Wrapper that launches the fused kernel
    M, K = x.shape
    K, N = weight.shape
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    linear_relu_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 128
    in_features = 768
    out_features = 3072
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    weight = torch.randn(in_features, out_features, device='cuda', dtype=torch.float16)
    bias = torch.randn(out_features, device='cuda', dtype=torch.float16)
    return (x, weight, bias)
# </TEST>
```

**examples/002_add_mul_add.py**
```python
"""Fused arithmetic operations: (x + y) * z + w"""

# <PYTHON>
def add_mul_add(x, y, z, w):
    # Three separate kernels:
    temp1 = x + y      # Kernel 1
    temp2 = temp1 * z  # Kernel 2  
    output = temp2 + w # Kernel 3
    return output
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def add_mul_add_kernel(x_ptr, y_ptr, z_ptr, w_ptr, output_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    # Load all inputs once
    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    z = tl.load(z_ptr + offset, mask=mask)
    w = tl.load(w_ptr + offset, mask=mask)
    
    # Fused computation - all in registers
    output = (x + y) * z + w
    
    # Single store
    tl.store(output_ptr + offset, output, mask=mask)

def add_mul_add(x, y, z, w):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_mul_add_kernel[grid](x, y, z, w, output, n_elements, BLOCK_SIZE=1024)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    size = 1000000
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    z = torch.randn(size, device='cuda')
    w = torch.randn(size, device='cuda')
    return (x, y, z, w)
# </TEST>
```

**examples/003_scaled_softmax.py**
```python
"""Scaled softmax for attention: softmax(x / sqrt(scale))"""

# <PYTHON>
def scaled_softmax(x, scale):
    # Multiple kernels:
    x = x / torch.sqrt(torch.tensor(scale))  # Kernel 1: div
    x_max = x.max(dim=-1, keepdim=True)[0]   # Kernel 2: max reduction
    x = x - x_max                             # Kernel 3: sub
    exp_x = torch.exp(x)                      # Kernel 4: exp
    sum_exp = exp_x.sum(dim=-1, keepdim=True) # Kernel 5: sum reduction
    return exp_x / sum_exp                     # Kernel 6: div
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl
import math

@triton.jit
def scaled_softmax_kernel(input_ptr, output_ptr, n_cols, scale,
                         BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols
    
    # Load entire row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
    
    # Scale
    x = x / tl.sqrt(scale)
    
    # Softmax - all fused
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp
    
    # Store
    tl.store(output_ptr + row_start + cols, softmax, mask=mask)

def scaled_softmax(x, scale):
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    scaled_softmax_kernel[grid](x, output, n_cols, float(scale), BLOCK_SIZE)
    return output
# </TRITON>

# <TEST>
def get_test_inputs():
    batch_size = 32
    seq_len = 512
    x = torch.randn(batch_size, seq_len, device='cuda')
    scale = 64.0  # sqrt(d_head) for attention
    return (x, scale)
# </TEST>
```

### AI Prompts for Generating Fused Examples

**Prompt Template:**
```
Generate a Triton example that FUSES multiple operations into a single kernel.

The Python version should show the operations as SEPARATE steps (multiple kernel launches).
The Triton version should FUSE them into ONE kernel.

Operations to fuse: [describe the sequence]

Example format:
# <PYTHON>
def function_name(...):
    # Show as separate operations with comments like:
    temp1 = operation1(x)     # Kernel 1: describe
    temp2 = operation2(temp1) # Kernel 2: describe
    return temp3
# </PYTHON>

# <TRITON>
[Single fused kernel that does all operations]
# </TRITON>

# <TEST>
[Realistic test inputs]
# </TEST>

Make sure the Triton kernel combines ALL operations into one kernel launch.
```

**Specific Prompts:**

1. **Activation Fusions:**
```
Generate fused kernel for: x * sigmoid(x) (SiLU/Swish activation)
Show Python as: sigmoid computation then multiplication (2 kernels)
```

2. **Normalization Fusions:**
```
Generate fused kernel for: layer_norm(x + residual)
Show Python as: addition then layer normalization (multiple kernels)
```

3. **Attention Components:**
```
Generate fused kernel for: softmax(QK^T / sqrt(d) + mask)
Show Python as: matmul, scale, add mask, softmax (4+ kernels)
```

4. **MLP Blocks:**
```
Generate fused kernel for: dropout(gelu(x @ W1 + b1)) @ W2 + b2
Show Python as separate matmul, add, activation, dropout, matmul, add
```

### Benchmark Script (Updated)

**scripts/benchmark.py**
```python
#!/usr/bin/env python3
import re
import torch
import time
import csv
from pathlib import Path

def count_kernel_launches(code):
    """Estimate kernel launches from Python code"""
    # Count operations that trigger kernels
    ops = ['torch.matmul', '@', '+', '-', '*', '/', 'torch.relu', 
           'torch.exp', 'torch.sigmoid', '.sum(', '.max(', '.mean(']
    return sum(1 for op in ops if op in code)

def benchmark_file(filepath):
    """Benchmark with kernel launch counting"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    python_code = re.search(r'# <PYTHON>(.*?)# </PYTHON>', content, re.DOTALL).group(1)
    triton_code = re.search(r'# <TRITON>(.*?)# </TRITON>', content, re.DOTALL).group(1)
    test_code = re.search(r'# <TEST>(.*?)# </TEST>', content, re.DOTALL).group(1)
    
    # Count approximate kernel launches
    kernel_launches_python = count_kernel_launches(python_code)
    kernel_launches_triton = 1  # Always 1 for fused kernel
    
    # Rest of benchmark...
    namespace = {'torch': torch, 'triton': __import__('triton'), 
                'tl': __import__('triton.language'), 'math': __import__('math')}
    
    exec(python_code, namespace)
    py_func = [v for k,v in namespace.items() if callable(v) and not k.startswith('_')][0]
    
    exec(triton_code, namespace)
    tr_func = [v for k,v in namespace.items() if callable(v) and not k.startswith('_') and v != py_func][-1]
    
    exec(test_code, namespace)
    inputs = namespace['get_test_inputs']()
    
    # Warmup
    for _ in range(10):
        py_func(*inputs)
        tr_func(*inputs)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        out_py = py_func(*inputs)
    torch.cuda.synchronize()
    py_time = time.perf_counter() - start
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        out_tr = tr_func(*inputs)
    torch.cuda.synchronize()
    tr_time = time.perf_counter() - start
    
    # Verify correctness
    correct = torch.allclose(out_py, out_tr, rtol=1e-3, atol=1e-3)
    
    return {
        'name': filepath.stem,
        'python_ms': py_time * 1000 / 100,
        'triton_ms': tr_time * 1000 / 100,
        'speedup': py_time / tr_time,
        'kernels_saved': kernel_launches_python - kernel_launches_triton,
        'correct': correct
    }

if __name__ == "__main__":
    results = []
    
    for file in sorted(Path("examples").glob("*.py")):
        print(f"Benchmarking {file.name}...", end=" ")
        result = benchmark_file(file)
        results.append(result)
        print(f"{result['speedup']:.2f}x speedup, "
              f"{result['kernels_saved']} kernels saved")
    
    # Save and summarize
    with open("output/benchmark_results.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    total_kernels_saved = sum(r['kernels_saved'] for r in results)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Total kernels saved: {total_kernels_saved}")
```

### Example Operations to Generate

```python
FUSED_OPERATIONS = [
    "linear + relu",
    "linear + gelu", 
    "x * sigmoid(x)",  # SiLU
    "(x + y) * z + w",
    "layer_norm(x + residual)",
    "softmax(x / sqrt(scale))",
    "dropout(relu(x @ W + b))",
    "x + dropout(x)",
    "mean((x - y) ** 2)",  # MSE
    "log(softmax(x))",  # Log softmax
    "x * (1 - x)",  # Derivative
    "tanh(x @ W1 + b1) @ W2 + b2",  # Small MLP
]
```

This approach focuses on **real performance gains** from kernel fusion, making the training data much more valuable for learning when and how to use Triton effectively.