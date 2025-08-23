# Distributed Training & Interactive Inference

This setup provides distributed training across multiple GPUs with periodic checkpointing and interactive text generation.

## 🚀 Features

- **Distributed Training**: Train on multiple GPUs using PyTorch DDP
- **Periodic Checkpointing**: Save model every 500 steps automatically
- **Interactive Inference**: Chat with your trained model
- **Centralized Data**: Single data loading server for all GPUs
- **Checkpoint Management**: Easy checkpoint listing and loading

## 📁 Files

- `distributed_train.py` - Multi-GPU training script
- `data_server.py` - Centralized data loading and caching
- `interactive_inference.py` - Interactive text generation
- `list_checkpoints.py` - List available checkpoints
- `run_distributed.sh` - Launch script for distributed training

## 🏃‍♂️ Quick Start

### 1. Start Distributed Training

```bash
# Make script executable
chmod +x run_distributed.sh

# Run distributed training
./run_distributed.sh
```

Or manually:
```bash
python distributed_train.py
```

**Requirements**: At least 2 GPUs for distributed training

### 2. Check Available Checkpoints

```bash
python list_checkpoints.py
```

### 3. Start Interactive Inference

```bash
# Use latest checkpoint (recommended)
python interactive_inference.py

# Use specific checkpoint
python interactive_inference.py --checkpoint checkpoints/checkpoint_step_1000.pt
```

## 🔧 Configuration

### Training Parameters

The training configuration is in `llm.py` in the `ModelConfig` class:

```python
@dataclass
class ModelConfig:
    d_model: int = 384          # Model dimension
    n_heads: int = 8            # Number of attention heads
    n_layers: int = 6           # Number of transformer layers
    d_ff: int = 1536            # Feed-forward dimension
    batch_size: int = 48        # Per-GPU batch size
    max_steps: int = 100        # Total training steps
```

### Checkpoint Frequency

Checkpoints are saved every **500 steps** by default. To change this, modify the line in `distributed_train.py`:

```python
# Save checkpoint every 500 steps
if rank == 0 and step % 500 == 0:
```

## 📊 Training Process

1. **Data Loading**: Central server loads and caches TheBlueScrubs medical dataset
2. **Model Distribution**: Same model replicated on each GPU
3. **Data Distribution**: Each GPU gets different batches via DistributedSampler
4. **Gradient Sync**: Gradients automatically averaged across GPUs
5. **Checkpointing**: Model saved every 500 steps to `checkpoints/` directory

## 🎭 Interactive Inference

### Commands

- `help` - Show available commands and parameters
- `clear` - Clear the screen
- `quit` - Exit the program

### Generation Parameters

- `--max-length 100` - Maximum tokens to generate
- `--temperature 0.8` - Creativity (higher = more random)
- `--top-k 50` - Top-k sampling for diversity
- `--top-p 0.9` - Nucleus sampling threshold

### Example Session

```bash
$ python interactive_inference.py

🔍 Using device: cuda
📚 Loading tokenizer...
📁 Found latest checkpoint: checkpoints/checkpoint_step_1000.pt (step 1000)
🔄 Loading model from checkpoints/checkpoint_step_1000.pt
✅ Model loaded successfully!
   Step: 1000
   Epoch: 2
   Loss: 2.3456

🎭 Interactive Text Generation Mode
Type 'quit' to exit, 'help' for options
==================================================

💬 Enter your prompt: The patient presents with

🤖 Generating text...
📝 Prompt: The patient presents with

✨ Generated text:
The patient presents with chest pain and shortness of breath. The symptoms began...
==================================================
```

## 🔍 Troubleshooting

### Common Issues

1. **"No checkpoints found"**
   - Run training first: `python distributed_train.py`
   - Check if `checkpoints/` directory exists

2. **CUDA out of memory**
   - Reduce `batch_size` in `ModelConfig`
   - Increase `gradient_accumulation_steps`

3. **Import errors**
   - Ensure all dependencies are installed
   - Check file paths and imports

### Performance Tips

- **GPU Utilization**: Monitor with `nvidia-smi`
- **Batch Size**: Increase for better GPU utilization
- **Checkpoint Frequency**: Reduce for more frequent saves (but slower training)

## 📈 Monitoring

### Training Progress

- Checkpoints saved every 500 steps
- Loss printed every 100 batches
- Model state saved with step, epoch, and loss

### Checkpoint Files

- Location: `checkpoints/checkpoint_step_X.pt`
- Contains: model state, optimizer state, config, metadata
- Size: ~50-100MB per checkpoint

## 🎯 Next Steps

1. **Train your model**: `python distributed_train.py`
2. **Monitor progress**: Check `checkpoints/` directory
3. **Test generation**: `python interactive_inference.py`
4. **Fine-tune**: Adjust hyperparameters in `ModelConfig`
5. **Scale up**: Increase model size or training steps

## 📚 Advanced Usage

### Custom Checkpoint Loading

```python
from interactive_inference import load_model_from_checkpoint

model, config = load_model_from_checkpoint("path/to/checkpoint.pt", device)
```

### Batch Generation

```python
from interactive_inference import generate_text

prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
for prompt in prompts:
    generated = generate_text(model, tokenizer, prompt)
    print(f"Generated: {generated}")
```

### Model Analysis

```python
# Get model parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Check model architecture
print(f"Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
```

Happy training and generating! 🚀
