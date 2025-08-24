# Instructions for AI-Powered Data Generation

**Objective**: Your primary function is to populate this repository with high-quality data examples of fused Triton kernels. You are a generative agent tasked with creating new `.py` files in the `triton_training_data/examples/` directory.

Your goal is to generate examples that, when tested with the `benchmark.py` script, are both **correct** and **demonstrate a performance speedup**, ultimately landing in the `triton_training_data/output/faster/` directory.

## Your Core Task: Creating New Examples

When asked to create a new example, you must generate a single Python file containing three distinct, specially-marked code blocks. You must strictly adhere to this format.

### 1. The `<PYTHON>` Block

- **Purpose**: To show the standard, unfused PyTorch implementation of a sequence of operations.
- **Your Task**: Write clear, simple PyTorch code. Use comments to highlight the individual operations that would each result in a separate GPU kernel launch (e.g., `y = x + z # Kernel 1: Add`). This illustrates the problem that Triton fusion solves.

### 2. The `<TRITON>` Block

- **Purpose**: To provide the optimized, fused Triton kernel that performs the same operations as the `<PYTHON>` block, but in a single GPU kernel launch.
- **Your Task**: This is the most critical part. You must write a correct and efficient Triton kernel. The kernel should combine all the intermediate steps from the Python version into one. This block must also include the Python wrapper function that launches the kernel.

### 3. The `<TEST>` Block

- **Purpose**: To provide the necessary inputs to run and verify both the Python and Triton implementations.
- **Your Task**: Write a function named `get_test_inputs()` that returns a tuple of Torch tensors. These tensors should be on the `'cuda'` device and have realistic shapes and data types (`float16`, `float32`) for the operation you are implementing.

## Your Workflow

1.  **Propose an Idea**: When asked, propose a new set of operations to fuse. Focus on common patterns in neural networks, such as activation functions, normalization layers, or parts of attention mechanisms.
2.  **Generate the File**: Create a new file in `triton_training_data/examples/` (e.g., `007_new_fusion.py`). Write the full, self-contained code for the `<PYTHON>`, `<TRITON>`, and `<TEST>` blocks.
3.  **Verification (Conceptual)**: Before finishing, you should mentally (or by analysis) verify your work. Does the Triton kernel correctly implement the Python logic? Are the tensor shapes in the test compatible with the operations? The ultimate test is the `benchmark.py` script.

## How to Succeed

Your success is measured by your ability to generate files that the `benchmark.py` script classifies as `faster`. This means your generated Triton code must be:

- **Correct**: It must produce a result that is numerically very close to the PyTorch version's output.
- **Performant**: It must run faster than the sequence of individual PyTorch kernels.

Strive to create examples that provide a significant and clear speedup. This repository is a testament to your ability to write optimized, low-level GPU code.