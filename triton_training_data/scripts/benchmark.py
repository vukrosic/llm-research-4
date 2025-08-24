#!/usr/bin/env python3
import re
import torch
import time
import shutil
from pathlib import Path
import tempfile
import importlib.util
import os

def benchmark_and_classify(filepath, faster_dir, slower_dir):
    """Benchmarks a single file and classifies it into faster/slower."""
    with open(filepath, 'r') as f:
        content = f.read()

    try:
        python_code = re.search(r'# <PYTHON>(.*?)# </PYTHON>', content, re.DOTALL).group(1)
        triton_code = re.search(r'# <TRITON>(.*?)# </TRITON>', content, re.DOTALL).group(1)
        test_code = re.search(r'# <TEST>(.*?)# </TEST>', content, re.DOTALL).group(1)
    except AttributeError:
        print(f"SKIPPING {filepath.name}: Could not parse file structure.")
        return

    # Prepare namespaces and function name
    py_namespace = {}
    exec(python_code, py_namespace)
    func_name = [k for k, v in py_namespace.items() if callable(v)][0]
    py_func = py_namespace[func_name]

    # Use temp file for Triton code to ensure inspect can find the source
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    tr_func = None
    try:
        temp_file.write(triton_code)
        temp_file.close()

        spec = importlib.util.spec_from_file_location("triton_module", temp_file.name)
        triton_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(triton_module)
        tr_func = getattr(triton_module, func_name)

        test_namespace = {}
        exec(test_code, test_namespace)
        inputs = test_namespace['get_test_inputs']()

        # Warmup and Benchmark
        for _ in range(10):
            py_func(*inputs)
            tr_func(*inputs)

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

        # Verification and Classification
        correct = torch.allclose(out_py, out_tr, rtol=1e-2, atol=1e-2)
        speedup = py_time / tr_time

        print(f"Speedup: {speedup:.2f}x, Correct: {'Yes' if correct else 'NO'}")

        if correct and speedup > 1.0:
            shutil.copy(filepath, faster_dir / filepath.name)
            print(f"CLASSIFIED as faster.")
        else:
            shutil.copy(filepath, slower_dir / filepath.name)
            print(f"CLASSIFIED as slower/incorrect.")

    except Exception as e:
        print(f"FAILED: {e}")
        shutil.copy(filepath, slower_dir / filepath.name)
        print(f"CLASSIFIED as slower/failed.")
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    examples_dir = base_dir / "examples"
    output_dir = base_dir / "output"

    faster_dir = output_dir / "faster"
    slower_dir = output_dir / "slower"

    # Clean up previous runs
    if faster_dir.exists():
        shutil.rmtree(faster_dir)
    if slower_dir.exists():
        shutil.rmtree(slower_dir)

    faster_dir.mkdir(exist_ok=True, parents=True)
    slower_dir.mkdir(exist_ok=True, parents=True)

    print(f"Classifying examples from: {examples_dir}")
    print(f"Faster examples will be saved to: {faster_dir}")
    print(f"Slower examples will be saved to: {slower_dir}\n")

    for file in sorted(examples_dir.glob("*.py")):
        print(f"--- Benchmarking {file.name} ---")
        benchmark_and_classify(file, faster_dir, slower_dir)
        print("---\n")

    print("Classification complete.")