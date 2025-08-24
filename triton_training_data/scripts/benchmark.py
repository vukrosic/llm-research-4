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
    
    for file in sorted(Path("triton_training_data/examples").glob("*.py")):
        print(f"Benchmarking {file.name}...", end=" ")
        result = benchmark_file(file)
        results.append(result)
        print(f"{result['speedup']:.2f}x speedup, "
              f"{result['kernels_saved']} kernels saved")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    total_kernels_saved = sum(r['kernels_saved'] for r in results)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")
    print(f"Total kernels saved: {total_kernels_saved}")
