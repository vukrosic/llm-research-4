import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import os
import time
from llm import MinimalLLM, ModelConfig, set_seed, setup_muon_optimizer, evaluate_model

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_worker(rank, world_size, config):
    """Training worker for each GPU"""
    setup_distributed(rank, world_size)
    
    # Load data
    from data_server import CentralDataServer
    server = CentralDataServer(config.num_documents, config.max_tokens, config.max_seq_len)
    data = server.load_and_cache_data()
    
    # Create dataset
    from llm import TextTokenDataset
    dataset = TextTokenDataset(data['tokens'], config.max_seq_len)
    
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
        batch_size=config.batch_size, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    config.vocab_size = data['vocab_size']
    model = MinimalLLM(config)
    model = model.to(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Setup optimizers
    optimizers = setup_muon_optimizer(model.module, config)
    
    # Training loop
    model.train()
    scaler = GradScaler() if config.use_amp else None
    
    if rank == 0:
        print(f"🚀 Starting distributed training on {world_size} GPUs")
        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config.max_steps // len(train_loader) + 1):
        train_sampler.set_epoch(epoch)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(rank), y.to(rank)
            
            # Forward pass
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, config.vocab_size), 
                        y.view(-1)
                    )
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size), 
                    y.view(-1)
                )
                loss.backward()
            
            # Optimizer step
            if config.use_amp:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    if rank == 0:
        print("✅ Training completed!")
    
    cleanup_distributed()

def main():
    """Main function to launch distributed training"""
    world_size = torch.cuda.device_count()
    print(f"�� Found {world_size} GPUs")
    
    if world_size < 2:
        print("❌ Need at least 2 GPUs for distributed training")
        return
    
    # Configuration
    config = ModelConfig()
    config.batch_size = 48  # Per-GPU batch size
    config.max_steps = 100
    
    print(f"📋 Distributed Training Configuration:")
    print(f"   World size: {world_size}")
    print(f"   Per-GPU batch size: {config.batch_size}")
    print(f"   Total batch size: {config.batch_size * world_size}")
    
    # Launch processes
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()