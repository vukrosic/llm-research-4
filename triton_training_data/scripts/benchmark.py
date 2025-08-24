#!/usr/bin/env python3
import re
import torch
import time
import shutil
from pathlib import Path
import tempfile
import importlib.util
import os
import sys
import json

def run_benchmark_on_file(filepath):
    """Runs the benchmark for a single file and returns a result dictionary."""
    with open(filepath, 'r') as f:
        content = f.read()

    # --- Parsing --- #
    try:
        python_code = re.search(r'# <PYTHON>(.*?)# </PYTHON>', content, re.DOTALL).group(1)
        triton_code = re.search(r'# <TRITON>(.*?)# </TRITON>', content, re.DOTALL).group(1)
        test_code = re.search(r'# <TEST>(.*?)# </TEST>', content, re.DOTALL).group(1)
    except AttributeError:
        return {"status": "FAILED", "reason": "Could not parse file structure."}

    # --- Execution --- #
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    try:
        py_namespace = {}
        exec(python_code, py_namespace)
        func_name = [k for k, v in py_namespace.items() if callable(v)][0]
        py_func = py_namespace[func_name]

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

        correct = torch.allclose(out_py, out_tr, rtol=1e-2, atol=1e-2)
        speedup = py_time / tr_time

        return {
            "status": "SUCCESS",
            "correct": correct,
            "speedup": speedup,
            "python_ms": py_time * 10,
            "triton_ms": tr_time * 10,
        }

    except Exception as e:
        return {"status": "FAILED", "reason": str(e)}
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

def classify_all_examples(base_dir):
    """Runs benchmark on all examples and classifies them."""
    examples_dir = base_dir / "examples"
    output_dir = base_dir / "output"
    faster_dir = output_dir / "faster"
    slower_dir = output_dir / "slower"

    if faster_dir.exists(): shutil.rmtree(faster_dir)
    if slower_dir.exists(): shutil.rmtree(slower_dir)
    faster_dir.mkdir(exist_ok=True, parents=True)
    slower_dir.mkdir(exist_ok=True, parents=True)

    print(f"Classifying examples from: {examples_dir}\n")
    for file in sorted(examples_dir.glob("*.py")):
        print(f"--- Benchmarking {file.name} ---")
        result = run_benchmark_on_file(file)
        if result["status"] == "SUCCESS" and result["correct"] and result["speedup"] > 1.0:
            print(f"RESULT: FASTER ({result['speedup']:.2f}x speedup)")
            shutil.copy(file, faster_dir / file.name)
        else:
            reason = result.get('reason', f"slower/incorrect (speedup: {result.get('speedup', 0):.2f}x, correct: {result.get('correct', False)})")
            print(f"RESULT: SLOWER or FAILED ({reason})")
            shutil.copy(file, slower_dir / file.name)
        print("---\n")
    print("Classification complete.")

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent

    # If a file path is provided as an argument, test it. Otherwise, classify all.
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        result = run_benchmark_on_file(filepath)
        print(json.dumps(result)) # Output machine-readable JSON
    else:
        classify_all_examples(base_dir)
