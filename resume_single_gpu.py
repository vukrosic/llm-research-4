import torch
import os
import argparse
import glob
from llm import MinimalLLM, ModelConfig, set_seed, setup_muon_optimizer, TextTokenDataset
from data_server import CentralDataServer

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None
    
    # Extract step numbers and find the latest
    step_numbers = []
    for file in checkpoint_files:
        try:
            step = int(file.split("_step_")[1].split(".pt")[0])
            step_numbers.append((step, file))
        except:
            continue
    
    if not step_numbers:
        return None
    
    # Return the latest checkpoint
    latest_step, latest_file = max(step_numbers, key=lambda x: x[0])
    return latest_step, latest_file

def resume_single_gpu(checkpoint_path, resume_config):
    """Resume training on single GPU"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Using device: {device}")
    
    # Load checkpoint
    print(f"🔄 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract training state
    start_step = checkpoint['step']
    start_epoch = checkpoint['epoch']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_states = checkpoint['optimizer_states']
    
    # Load data
    server = CentralDataServer(resume_config.num_documents, resume_config.max_tokens, resume_config.max_seq_len)
    data = server.load_and_cache_data()
    
    # Create dataset
    dataset = TextTokenDataset(data['tokens'], resume_config.max_seq_len)
    
    # Split data
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=resume_config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=resume_config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Create model
    resume_config.vocab_size = data['vocab_size']
    model = MinimalLLM(resume_config)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    
    # Setup optimizers
    optimizers = setup_muon_optimizer(model, resume_config)
    
    # Load optimizer states if available and compatible
    if len(optimizer_states) == len(optimizers):
        try:
            for opt, opt_state in zip(optimizers, optimizer_states):
                opt.load_state_dict(opt_state)
            print(f"✅ Loaded optimizer states successfully")
        except Exception as e:
            print(f"⚠️ Could not load optimizer states: {e}")
            print(f"   Starting with fresh optimizers")
    
    # Training loop
    model.train()
    
    print(f"🚀 Resuming training on single GPU")
    print(f"📊 Starting from step {start_step}, epoch {start_epoch}")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    step = start_step
    for epoch in range(start_epoch, resume_config.max_steps // len(train_loader) + 1):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, resume_config.vocab_size), 
                y.view(-1)
            )
            loss.backward()
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), resume_config.grad_clip)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Step {step}, Loss: {loss.item():.4f}")
            
            # Save checkpoint every 500 steps
            if step % 500 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_states': [opt.state_dict() for opt in optimizers],
                    'config': resume_config,
                    'loss': loss.item(),
                    'resumed_from': checkpoint_path,
                }, checkpoint_path)
                print(f"💾 Saved checkpoint to {checkpoint_path}")
            
            # Check if we've reached max steps
            if step >= resume_config.max_steps:
                break
        
        if step >= resume_config.max_steps:
            break
    
    print("✅ Training completed!")

def main():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint on single GPU")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file (auto-finds latest if not specified)")
    parser.add_argument("--max-steps", type=int, help="Total training steps (continues from checkpoint step)")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        latest_step, latest_file = find_latest_checkpoint()
        if latest_file:
            print(f"📁 Latest checkpoint: {latest_file} (step {latest_step})")
        else:
            print("❌ No checkpoints found!")
        return
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
    else:
        latest_step, latest_file = find_latest_checkpoint()
        if not latest_file:
            print("❌ No checkpoints found! Please train the model first.")
            print("💡 Run: python distributed_train.py")
            return
        checkpoint_path = latest_file
        print(f"📁 Using latest checkpoint: {checkpoint_path}")
    
    # Load checkpoint to get config
    print(f"🔄 Loading checkpoint to get configuration...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    original_config = checkpoint['config']
    
    # Create resume config (can override some parameters)
    resume_config = ModelConfig()
    
    # Override with checkpoint config
    for key, value in vars(original_config).items():
        if hasattr(resume_config, key):
            setattr(resume_config, key, value)
    
    # Override with command line arguments
    if args.max_steps:
        resume_config.max_steps = args.max_steps
    if args.batch_size:
        resume_config.batch_size = args.batch_size
    
    # Show resume configuration
    print(f"\n📋 Resume Training Configuration:")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Starting step: {checkpoint['step']}")
    print(f"   Starting epoch: {checkpoint['epoch']}")
    print(f"   Target steps: {resume_config.max_steps}")
    print(f"   Batch size: {resume_config.batch_size}")
    
    # Confirm before starting
    response = input("\n🚀 Continue with these settings? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled")
        return
    
    # Start training
    resume_single_gpu(checkpoint_path, resume_config)

if __name__ == "__main__":
    main()
