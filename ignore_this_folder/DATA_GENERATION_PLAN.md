# Plan for Generating Python-to-Triton Dataset

## 1. Project Goal

The primary objective is to create a comprehensive and high-quality dataset to train a Large Language Model (LLM). This model will be specialized in translating Python code, particularly functions involving tensor operations, into highly optimized Triton kernels. The dataset will consist of Python function-Triton kernel pairs and corresponding tests to ensure functional equivalence.

## 2. Dataset Structure

Each entry in the dataset will be a JSON object with the following structure:

```json
{
  "id": "unique_identifier_for_the_entry",
  "python_code": "source_code_of_the_python_function",
  "triton_code": "source_code_of_the_equivalent_triton_kernel",
  "test_code": "a_pytest_function_to_verify_correctness",
  "metadata": {
    "operation_type": "e.g., element-wise, reduction, matmul",
    "difficulty": "e.g., easy, medium, hard",
    "tags": ["e.g.", "vector-addition", "softmax"]
  }
}
```

## 3. Data Generation Strategy

We will follow a phased approach to generate the data, starting with simple operations and gradually increasing complexity.

### Phase 1: Foundational Kernels (Manual Curation)

- **Objective**: Create a small, high-quality set of foundational examples.
- **Process**:
    1. Manually write Python functions for basic tensor operations.
    2. Manually write the corresponding Triton kernels.
    3. Manually write Pytest functions to compare the outputs of the Python and Triton functions.
- **Operations to Cover**:
    - **Element-wise**: Vector addition, subtraction, multiplication, division.
    - **Reductions**: Sum, max, min.
    - **Matrix Operations**: Matrix multiplication (Matmul).

### Phase 2: Automated Generation (Templating)

- **Objective**: Scale the dataset by automating the generation of variations.
- **Process**:
    1. Develop a Python script that uses a templating engine (e.g., Jinja2) to generate code.
    2. Create templates for Python functions, Triton kernels, and Pytest checks.
    3. The script will generate numerous variations by changing:
        - Tensor shapes and sizes.
        - Data types (e.g., `float32`, `float16`, `bfloat16`).
        - Function and variable names.

### Phase 3: Complex and Real-World Kernels

- **Objective**: Incorporate more complex and realistic examples.
- **Process**:
    1. Source examples from open-source projects, research papers, and tutorials.
    2. Implement more complex algorithms:
        - Attention mechanisms (e.g., FlashAttention).
        - Convolutions.
        - Layer normalization.
    3. For each complex example, create the Python-Triton pair and the corresponding test.

## 4. Testing and Validation

- **Framework**: Pytest will be used for all testing.
- **Methodology**:
    1. For each data entry, the `test_code` will be executed.
    2. The test will generate random input tensors.
    3. It will run both the Python and Triton functions with the same inputs.
    4. The outputs will be compared using `torch.allclose` to ensure they are numerically close, accounting for potential floating-point inaccuracies.

## 5. Tools and Libraries

- **Programming Language**: Python
- **Core Libraries**:
    - **PyTorch**: For tensor operations and as the foundation for Triton.
    - **Triton**: The target language for the kernels.
    - **Pytest**: For testing and validation.
    - **NumPy**: For numerical operations and data generation.
    - **Jinja2**: For code templating in Phase 2.

## 6. Proposed Directory Structure

The project will be organized as follows:

```
/
├── data_generation/
│   ├── generate.py
│   ├── templates/
│   │   ├── python/
│   │   ├── triton/
│   │   └── pytest/
│   └── utils.py
├── dataset/
│   ├── train/
│   ├── test/
│   └── validation/
└── ... (existing files)
```

## 7. Next Steps

1. Create the `data_generation` directory and the initial file structure.
2. Begin implementing Phase 1 by manually creating the first few examples of element-wise operations.
3. Develop the initial version of the `generate.py` script.
