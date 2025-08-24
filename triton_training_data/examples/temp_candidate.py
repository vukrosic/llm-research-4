"""Fused Adam Optimizer Update."""

# <PYTHON>
import torch

def adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    # Standard Adam update, many small kernels
    m.mul_(beta1).addcmul_(grad, torch.tensor(1.0 - beta1))
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    
    update = lr * m_hat / (torch.sqrt(v_hat) + eps)
    param.sub_(update)
    return param, m, v
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, t,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Load data
    param = tl.load(param_ptr + offset, mask=mask)
    grad = tl.load(grad_ptr + offset, mask=mask)
    m = tl.load(m_ptr + offset, mask=mask)
    v = tl.load(v_ptr + offset, mask=mask)

    # Fused update logic
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * grad * grad

    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)

    update = lr * m_hat / (tl.sqrt(v_hat) + eps)
    param_new = param - update

    # Store updated values
    tl.store(param_ptr + offset, param_new, mask=mask)
    tl.store(m_ptr + offset, m_new, mask=mask)
    tl.store(v_ptr + offset, v_new, mask=mask)

def adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    n_elements = param.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adam_kernel[grid](
        param, grad, m, v,
        lr, beta1, beta2, eps, t,
        n_elements, BLOCK_SIZE=1024
    )
    return param, m, v
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    size = 1024 * 1024
    param = torch.randn(size, device='cuda', dtype=torch.float32)
    grad = torch.randn(size, device='cuda', dtype=torch.float32) * 0.1
    m = torch.zeros(size, device='cuda', dtype=torch.float32)
    v = torch.zeros(size, device='cuda', dtype=torch.float32)
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 10 # Timestep
    # Return copies for the python version so the triton version doesn't modify them in place
    return (param.clone(), grad.clone(), m.clone(), v.clone(), lr, beta1, beta2, eps, t)
# </TEST>
