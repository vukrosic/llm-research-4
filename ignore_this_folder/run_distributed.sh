#!/bin/bash

# Script to run distributed training
echo "🚀 Starting distributed training on $(nvidia-smi --list-gpus | wc -l) GPUs"

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Run distributed training
python distributed_train.py

echo "✅ Distributed training completed!"