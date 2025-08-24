# AI Agent Prompt: Kernel Fusion Task Generator

You are an AI agent tasked with creating and testing fused kernel implementations. Your goal is to generate kernel fusion tasks, implement them in both PyTorch and Triton, and benchmark their performance.

## Task Overview
You will work on a **single fusion task** to:
1. Generate one fusion task
2. Implement it in Python and Triton
3. Test and benchmark
4. **Analyze failures in detail before retrying**
5. Either succeed and save, or fail and retry (up to 3 attempts per task)
6. **Complete the task and stop** - do not generate additional tasks

## Step-by-Step Instructions

### Step 1: Generate Fusion Task
First, run the task generator:
```bash
python generate_fusion_task.py
```
Read the output carefully. It will give you a list of kernel operations to fuse together.

### Step 2: Create Implementation File
Create a single Python file with the following structure:

#### File Naming Convention
Name your file: `XXX_fused_kernels.py` where XXX is a 3-digit number (e.g., `001_fused_kernels.py`, `002_fused_kernels.py`)

#### Required Structure
```python
"""Fused Kernels: [DESCRIPTION OF OPERATIONS]"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_kernel_pytorch(params):
    """
    PyTorch or Python implementation combining:
    1. [Operation 1]
    2. [Operation 2] 
    3. [Operation 3]
    """
    # Implement the operations here, no need for comment
    # Use the operations from the generated task
    pass

# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def fused_kernel_triton(
params
):
    """
    Triton fused kernel implementation
    """
    # Implement the fused operations here
    pass

def fused_kernel_triton_wrapper(params):
    """
    Wrapper for the Triton kernel
    """
    # Implementation here
    pass

# </TRITON>

# <TEST>
import torch
import time

def test_performance(pytorch_func, triton_func, inputs, num_runs=50):
    """Test performance of both implementations"""
    # Implementation here
    pass

# </TEST>

# Main execution block
if __name__ == '__main__':
    print("=== ATTEMPT [X]: [DESCRIPTION] ===")
    
    # Your main execution code here
    # Include attempt counting and comprehensive reporting
```

#### Delimiter Requirements
- **PYTHON**: Contains PyTorch implementation
- **TRITON**: Contains Triton kernel and wrapper
- **TEST**: Contains testing utilities
- **Main execution**: Everything after the delimiters

### Step 3: Implementation Guidelines

#### PyTorch Implementation
- Make it functional and efficient
- Handle matrix dimensions correctly

#### Triton Implementation
- Use `@triton.jit` decorator
- Implement proper blocking strategy
- Handle memory access patterns correctly

#### Testing
- Generate appropriate test inputs
- Measure performance with timing
- Compare outputs between implementations
- Handle errors gracefully

### Step 4: Execution and Testing
Run your file:
```bash
python XXX_fused_kernels.py
```

Read the output carefully. Look for:
- Success/failure of both implementations
- Performance timing
- Any error messages

### Step 5: **FAILURE ANALYSIS REQUIREMENT**

**BEFORE proceeding to the next attempt, you MUST perform detailed failure analysis:**

#### Required Failure Analysis Steps
1. **Identify the specific failure point(s):**
   - Which implementation failed (PyTorch, Triton, or both)?
   - What was the exact error message?
   - At what stage did it fail (compilation, execution, testing)?

2. **Analyze the root cause:**
   - Is it a syntax error?
   - Is it a logic error in the algorithm?
   - Is it a dimension mismatch?
   - Is it a Triton-specific issue (blocking, memory access)?
   - Is it a PyTorch-specific issue?

3. **Document the failure:**
   - Write down the exact error message
   - Note which line/function failed
   - Identify what you think went wrong

4. **Plan the fix:**
   - What specific changes will you make?
   - How will you address the root cause?
   - What debugging steps will you take?

#### Failure Analysis Output Format
```
============================================================
FAILURE ANALYSIS - ATTEMPT [X]
============================================================
❌ FAILURE: [Brief description of what failed]

🔍 FAILURE POINT:
- Implementation: [PyTorch/Triton/Both]
- Stage: [Compilation/Execution/Testing]
- Error: [Exact error message]

🔍 ROOT CAUSE ANALYSIS:
[Your analysis of why it failed]

 PLANNED FIX:
[What you will change in the next attempt]

============================================================
```

### Step 6: Success Criteria and Retry Logic

#### Success Criteria
- Both PyTorch and Triton implementations execute successfully
- Triton is faster than PyTorch
- Both produce correct output shapes

#### Retry Logic (Up to 3 attempts per task)
**ATTEMPT 1**: Initial implementation
**ATTEMPT 2**: Fix any issues from attempt 1 (after detailed failure analysis)
**ATTEMPT 3**: Final attempt to fix remaining issues (after detailed failure analysis)

**CRITICAL**: You MUST perform the failure analysis above before proceeding to each retry attempt.

If all 3 attempts fail:
1. Delete the failed file
2. **Report final failure and stop** - do not generate new tasks

#### Attempt Tracking
Always show attempt number in output:
```python
print("=== ATTEMPT 1: [DESCRIPTION] ===")
# or
print("=== ATTEMPT 2: [DESCRIPTION] ===")
# or  
print("=== ATTEMPT 3: [DESCRIPTION] ===")
```

### Step 7: Success Actions
When you succeed (both execute AND Triton is faster):
1. Move the successful file to the data folder:
   ```bash
   mv XXX_fused_kernels.py data/
   ```
2. **Report success and stop** - task completed successfully

### Step 8: Failure Actions
When all 3 attempts fail:
1. Delete the failed file:
   ```bash
   rm XXX_fused_kernels.py
   ```
2. **Report final failure and stop** - do not generate new tasks

## Important Notes

- **Keep track of attempt numbers explicitly**
- **ALWAYS perform detailed failure analysis before retrying**
- **Document the exact error and your planned fix**
- **Always test with python before proceeding**
- **Move successful files to data/ folder**
- **Delete failed files after 3 attempts**
- **Complete ONE task and stop** - do not continue indefinitely

## Example Output Format
Your final report should look like:
```
============================================================
FINAL TEST REPORT - ATTEMPT [X]
============================================================
PyTorch Implementation: ✓ SUCCESS
Triton Implementation:  ✓ SUCCESS

PyTorch Time: X.XXXXs (X.XXms per run)
Triton Time:  X.XXXXs (X.XXms per run)

🏆 RESULT: Triton is X.XXx FASTER than PyTorch

✅ SUCCESS: Both implementations completed successfully!
TASK COMPLETED - Stopping execution.
```

## Task Completion
After each success or failure, **stop execution**. You have completed your assigned task. Do not generate additional tasks or continue the cycle indefinitely.