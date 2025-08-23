#!/usr/bin/env python3
"""
Simple script to list all available checkpoints
"""

import os
import glob
from datetime import datetime

def list_checkpoints(checkpoint_dir="checkpoints"):
    """List all available checkpoints with details"""
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoints directory found!")
        print("💡 Run training first: python distributed_train.py")
        return
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        print("💡 Run training first: python distributed_train.py")
        return
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint(s) in {checkpoint_dir}/")
    print("=" * 60)
    
    # Extract step numbers and file info
    checkpoint_info = []
    for file in checkpoint_files:
        try:
            step = int(file.split("_step_")[1].split(".pt")[0])
            file_size = os.path.getsize(file) / (1024 * 1024)  # MB
            mod_time = os.path.getmtime(file)
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            
            checkpoint_info.append((step, file, file_size, mod_time_str))
        except:
            continue
    
    # Sort by step number
    checkpoint_info.sort(key=lambda x: x[0])
    
    print(f"{'Step':<8} {'Size (MB)':<10} {'Modified':<20} {'File'}")
    print("-" * 60)
    
    for step, file, size, mod_time in checkpoint_info:
        filename = os.path.basename(file)
        print(f"{step:<8} {size:<10.1f} {mod_time:<20} {filename}")
    
    print("=" * 60)
    
    if checkpoint_info:
        latest_step = max(checkpoint_info, key=lambda x: x[0])[0]
        print(f"🎯 Latest checkpoint: step {latest_step}")
        print(f"💡 To use latest: python interactive_inference.py")
        print(f"💡 To use specific: python interactive_inference.py --checkpoint checkpoints/checkpoint_step_{latest_step}.pt")

if __name__ == "__main__":
    list_checkpoints()
