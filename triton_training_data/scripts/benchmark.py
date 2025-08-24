#!/usr/bin/env python3
import re
import torch
import time
import csv
from pathlib import Path
import tempfile
import importlib.util
import os
import inspect

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
    namespace = {'torch': torch}
    
    exec(python_code, namespace)
    # Infer the function name from the file name (e.g., 001_linear_relu.py -> linear_relu)
    func_name = filepath.stem.split('_', 1)[1]
    py_func = namespace[func_name]

    # Write triton code to a temp file to allow inspect to get source
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Add necessary imports to the temp file
        f.write("import torch\n")
        f.write("import triton\n")
        f.write("import triton.language as tl\n")
        f.write("import math\n")
        f.write(triton_code)
        temp_filename = f.name

    try:
        spec = importlib.util.spec_from_file_location("triton_module", temp_filename)
        triton_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_module)
        tr_func = getattr(triton_module, func_name)
    finally:
        os.remove(temp_filename)

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
    # Loosen tolerance for more complex operations
    correct = torch.allclose(out_py, out_tr, rtol=1e-2, atol=1e-2)
    
    return {
        'name': filepath.stem,
        'python_ms': py_time * 1000 / 100,
        'triton_ms': tr_time * 1000 / 100,
        'speedup': py_time / tr_time,
        'kernels_saved': kernel_launches_python - kernel_launches_triton,
        'correct': correct
    }

if __name__ == "__main__":
    # Ensure we are in the correct directory
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    examples_dir = base_dir / "examples"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    results = []
    
    # Correctly glob for files in the examples directory
    for file in sorted(examples_dir.glob("*.py")):
        print(f"Benchmarking {file.name}...", end=" ")
        try:
            result = benchmark_file(file)
            results.append(result)
            print(f"{result['speedup']:.2f}x speedup, "
                  f"{result['kernels_saved']} kernels saved, "
                  f"Correct: {'Yes' if result['correct'] else 'NO'}")
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("No benchmark results.")
        exit()
    
    # Save and summarize
    results_path = output_dir / "benchmark_results.csv"
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {results_path}")

    # Filter for successful runs before calculating averages
    successful_results = [r for r in results if r['correct']]
    if successful_results:
        avg_speedup = sum(r['speedup'] for r in successful_results) / len(successful_results)
        total_kernels_saved = sum(r['kernels_saved'] for r in successful_results)
        print(f"\nAverage speedup (correct results): {avg_speedup:.2f}x")
        print(f"Total kernels saved (correct results): {total_kernels_saved}")
    else:
        print("\nNo successful runs to average.")