import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import os
import time
import argparse
import glob
from ignore_this_folder.llm import MinimalLLM, ModelConfig, set_seed, setup_muon_optimizer, evaluate_model, TextTokenDataset
from data_server import CentralDataServer

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

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

def resume_worker(rank, world_size, checkpoint_path, resume_config):
    """Resume training worker for each GPU"""
    setup_distributed(rank, world_size)
    
    # Load checkpoint
    print(f"🔄 Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}', weights_only=False)
    except Exception as e:
        print(f"⚠️ Standard loading failed: {e}")
        print("🔄 Trying with weights_only=False for PyTorch 2.6+ compatibility...")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}', weights_only=False)
    
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
    
    # Split data for distributed training
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=resume_config.batch_size, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=resume_config.batch_size, 
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    resume_config.vocab_size = data['vocab_size']
    model = MinimalLLM(resume_config)
    model.load_state_dict(model_state_dict)
    model = model.to(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Setup optimizers
    optimizers = setup_muon_optimizer(model.module, resume_config)
    
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
    scaler = GradScaler() if resume_config.use_amp else None
    
    if rank == 0:
        print(f"🚀 Resuming distributed training on {world_size} GPUs")
        print(f"📊 Starting from step {start_step}, epoch {start_epoch}")
        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Checkpoint directory
    checkpoint_dir = "checkpoints"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    step = start_step
    for epoch in range(start_epoch, resume_config.max_steps // len(train_loader) + 1):
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(rank), y.to(rank)
            
            # Forward pass
            if resume_config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, resume_config.vocab_size), 
                        y.view(-1)
                    )
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, resume_config.vocab_size), 
                    y.view(-1)
                )
                loss.backward()
            
            # Optimizer step
            if resume_config.use_amp:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), resume_config.grad_clip)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resume_config.grad_clip)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            
            step += 1
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Step {step}, Loss: {loss.item():.4f}")
            
            # Save checkpoint every 500 steps
            if rank == 0 and step % 500 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_states': [opt.state_dict() for opt in optimizers],
                    'config': resume_config,
                    'loss': loss.item(),
                    'resumed_from': checkpoint_path,  # Track resume history
                }, checkpoint_path, _use_new_zipfile_serialization=False)
                print(f"💾 Saved checkpoint to {checkpoint_path}")
            
            # Check if we've reached max steps
            if step >= resume_config.max_steps:
                break
        
        if step >= resume_config.max_steps:
            break
    
    if rank == 0:
        print("✅ Training completed!")
    
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Resume distributed training from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file (auto-finds latest if not specified)")
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use (auto-detects if not specified)")
    parser.add_argument("--max-steps", type=int, help="Total training steps (continues from checkpoint step)")
    parser.add_argument("--batch-size", type=int, help="Per-GPU batch size")
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
    try:
        # Try loading with weights_only=False for PyTorch 2.6+ compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"⚠️ Standard loading failed: {e}")
        print("🔄 Trying with weights_only=False for PyTorch 2.6+ compatibility...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
    
    # GPU setup
    if args.gpus:
        world_size = args.gpus
        if world_size > torch.cuda.device_count():
            print(f"❌ Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
            return
    else:
        world_size = torch.cuda.device_count()
    
    print(f"🔍 Using {world_size} GPUs")
    
    if world_size < 1:
        print("❌ No GPUs available!")
        return
    
    # Show resume configuration
    print(f"\n📋 Resume Training Configuration:")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Starting step: {checkpoint['step']}")
    print(f"   Starting epoch: {checkpoint['epoch']}")
    print(f"   Target steps: {resume_config.max_steps}")
    print(f"   World size: {world_size}")
    print(f"   Per-GPU batch size: {resume_config.batch_size}")
    print(f"   Total batch size: {resume_config.batch_size * world_size}")
    
    # Confirm before starting
    response = input("\n🚀 Continue with these settings? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled")
        return
    
    # Launch processes
    if world_size == 1:
        # Single GPU - no need for distributed
        print("🔄 Single GPU mode - no distributed training needed")
        # You could implement single GPU resume here
        print("💡 For single GPU, consider using the original llm.py with checkpoint loading")
    else:
        # Multi-GPU distributed training
        print(f"🚀 Launching distributed training on {world_size} GPUs...")
        mp.spawn(
            resume_worker,
            args=(world_size, checkpoint_path, resume_config),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    main()
