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
    
    # Wrap with
