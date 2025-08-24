#!/usr/bin/env python3
"""
Test script to diagnose checkpoint loading issues
"""

import torch
import os
import glob
import sys

def test_checkpoint_loading(checkpoint_path):
    """Test loading a checkpoint with different methods"""
    print(f"🔍 Testing checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Method 1: Try standard loading
    print("\n📥 Method 1: Standard loading...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Standard loading successful!")
        return True
    except Exception as e:
        print(f"❌ Standard loading failed: {e}")
    
    # Method 2: Try with weights_only=False
    print("\n📥 Method 2: Loading with weights_only=False...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Loading with weights_only=False successful!")
        return True
    except Exception as e:
        print(f"❌ Loading with weights_only=False failed: {e}")
    
    # Method 3: Try with pickle_module
    print("\n📥 Method 3: Loading with pickle_module...")
    try:
        import pickle
        checkpoint = torch.load(checkpoint_path, map_location='cpu', pickle_module=pickle)
        print("✅ Loading with pickle_module successful!")
        return True
    except Exception as e:
        print(f"❌ Loading with pickle_module failed: {e}")
    
    # Method 4: Try to load just the model weights
    print("\n📥 Method 4: Loading just model weights...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("✅ Loading just weights successful!")
        print("⚠️ Note: This only loads model weights, not config or optimizer states")
        return True
    except Exception as e:
        print(f"❌ Loading just weights failed: {e}")
    
    return False

def list_checkpoints():
    """List available checkpoints"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"❌ No checkpoints directory found!")
        return []
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}/")
        return []
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint(s):")
    for i, file in enumerate(checkpoint_files):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   {i+1}. {os.path.basename(file)} ({size_mb:.1f} MB)")
    
    return checkpoint_files

def main():
    print("🔍 Checkpoint Loading Test Script")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💾 CUDA available: {torch.cuda.is_available()}")
    
    # List checkpoints
    checkpoint_files = list_checkpoints()
    
    if not checkpoint_files:
        print("\n💡 No checkpoints to test. Run training first:")
        print("   python distributed_train.py")
        return
    
    # Test latest checkpoint
    latest_checkpoint = checkpoint_files[-1]
    print(f"\n🧪 Testing latest checkpoint: {os.path.basename(latest_checkpoint)}")
    
    success = test_checkpoint_loading(latest_checkpoint)
    
    if success:
        print(f"\n✅ Checkpoint loading successful!")
        print(f"💡 You can now use resume training scripts:")
        print(f"   python resume_training.py")
        print(f"   python resume_single_gpu.py")
    else:
        print(f"\n❌ Checkpoint loading failed!")
        print(f"💡 This indicates a compatibility issue.")
        print(f"💡 Try updating the resume scripts or check PyTorch version compatibility.")

if __name__ == "__main__":
    main()
