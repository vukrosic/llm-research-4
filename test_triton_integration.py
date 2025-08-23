#!/usr/bin/env python3
"""
Test script to verify Triton integration in llm.py
"""

import torch
import torch.nn as nn
from llm import ModelConfig, MinimalLLM, auto_select_kernels, manual_kernel_benchmark

def test_triton_integration():
    """Test that Triton kernels are properly integrated"""
    print("🧪 Testing Triton Integration")
    print("=" * 40)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test Triton kernels.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    # Create a small config for testing
    config = ModelConfig()
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    config.max_seq_len = 64
    config.batch_size = 4
    config.max_steps = 10
    
    print(f"📋 Test config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    try:
        # Test auto-selection
        print("\n🔍 Testing kernel auto-selection...")
        kernel_selection = auto_select_kernels(config)
        print(f"✅ Auto-selection completed:")
        print(f"   RMSNorm: {'Triton' if kernel_selection['use_triton_rmsnorm'] else 'PyTorch'}")
        print(f"   Rotary:  {'Triton' if kernel_selection['use_triton_rotary'] else 'PyTorch'}")
        
        # Test model creation
        print("\n🔍 Testing model creation...")
        model = MinimalLLM(config)
        model = model.to('cuda')
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        print("\n🔍 Testing forward pass...")
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device='cuda')
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass successful: {output.shape}")
        
        # Test manual benchmark
        print("\n🔍 Testing manual benchmark...")
        manual_kernel_benchmark()
        
        print("\n🎉 All tests passed! Triton integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_triton_integration()
    if success:
        print("\n🚀 Triton integration is ready for training!")
    else:
        print("\n⚠️  Triton integration has issues. Check the error messages above.")
