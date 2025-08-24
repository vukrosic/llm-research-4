"""Fused cross-entropy loss with label smoothing"""

# <PYTHON>
import torch
import torch.nn.functional as F

def fused_cross_entropy_loss(logits, targets, label_smoothing=0.1):
    # Multiple operations in PyTorch:
    log_probs = F.log_softmax(logits, dim=-1)    # Kernel 1: softmax then log
    loss = F.nll_loss(log_probs, targets, reduction='none')  # Kernel 2: negative log likelihood
    
    # Label smoothing (additional operations)
    if label_smoothing > 0:
        n_classes = logits.shape[-1]
        smooth_loss = -log_probs.mean(dim=-1)    # Kernel 3: mean
        loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss  # Kernel 4: scale and add
    
    return loss.mean()  # Kernel 5: mean
# </PYTHON>

# <TRITON>
import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_loss_kernel(
    logits_ptr, targets_ptr, loss_ptr,
    n_cols, n_rows,
    label_smoothing,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    row_logits_ptr = logits_ptr + row_idx * n_cols
    targets_ptr_row = targets_ptr + row_idx
    
    # Load target
    target = tl.load(targets_ptr_row).to(tl.int64)
    
    # Load logits for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(row_logits_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
    
    # Compute log softmax in one go
    max_logits = tl.max(logits, axis=0)
    shifted = logits - max_logits
    exp_logits = tl.exp(shifted)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp_logits) + max_logits
    log_probs = logits - log_sum_exp
    
    # Compute NLL loss
    target_log_prob = tl.load(row_logits_ptr + target) - log_sum_exp
    loss = -target_log_prob
    
    # Apply label smoothing if needed
    if label_smoothing > 0.0:
        mean_log_prob = -tl.sum(log_probs * (exp_logits / sum_exp_logits), axis=0)
        loss = (1 - label_smoothing) * loss + label_smoothing * mean_log_prob
    
    # Store result
    tl.store(loss_ptr + row_idx, loss)

def fused_cross_entropy_loss(logits, targets, label_smoothing=0.1):
    n_rows, n_cols = logits.shape
    loss = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)
    
    grid = (n_rows,)
    cross_entropy_loss_kernel[grid](
        logits, targets, loss,
        n_cols, n_rows,
        label_smoothing,
        BLOCK_SIZE=triton.next_power_of_2(n_cols)
    )
    
    return torch.sum(loss) / n_rows
# </TRITON>

# <TEST>
import torch

def get_test_inputs():
    batch_size = 64
    vocab_size = 32000
    logits = torch.randn(batch_size, vocab_size, device='cuda', dtype=torch.float16)
    targets = torch.randint(0, vocab_size, (batch_size,), device='cuda')
    label_smoothing = 0.1
    return (logits, targets, label_smoothing)
# </TEST>