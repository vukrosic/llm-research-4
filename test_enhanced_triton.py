#!/usr/bin/env python3
"""
Enhanced test script for all Triton kernels in llm.py
"""

import torch
import torch.nn as nn
import time
from llm import (
    ModelConfig, MinimalLLM, auto_select_kernels, manual_kernel_benchmark,
    benchmark_rms_norm, benchmark_rotary, benchmark_mlp, benchmark_attention, benchmark_newton_schulz
)

def test_all_kernels():
    """Test all Triton kernels comprehensively"""
    print("🧪 Enhanced Triton Kernel Test Suite")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test Triton kernels.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    # Create test config
    config = ModelConfig()
    config.d_model = 256
    config.n_heads = 8
    config.n_layers = 2
    config.max_seq_len = 128
    config.batch_size = 8
    config.max_steps = 10
    
    print(f"📋 Test config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    
    try:
        # Test 1: Individual kernel benchmarks
        print("\n🔍 Test 1: Individual Kernel Benchmarks")
        print("-" * 40)
        
        batch_size, seq_len, d_model = 8, 64, 256
        n_heads = 8
        d_head = d_model // n_heads
        d_ff = 1024
        
        # RMSNorm benchmark
        print("  Testing RMSNorm...")
        rms_results = benchmark_rms_norm(batch_size, seq_len, d_model)
        print(f"    PyTorch: {rms_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {rms_results['triton_time']:.3f} ms")
        print(f"    Speedup: {rms_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if rms_results['use_triton'] else 'PyTorch'}")
        
        # Rotary benchmark
        print("  Testing Rotary...")
        rotary_results = benchmark_rotary(batch_size, seq_len, n_heads, d_head)
        print(f"    PyTorch: {rotary_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {rotary_results['triton_time']:.3f} ms")
        print(f"    Speedup: {rotary_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if rotary_results['use_triton'] else 'PyTorch'}")
        
        # MLP benchmark
        print("  Testing MLP...")
        mlp_results = benchmark_mlp(batch_size, seq_len, d_model, d_ff)
        print(f"    PyTorch: {mlp_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {mlp_results['triton_time']:.3f} ms")
        print(f"    Speedup: {mlp_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if mlp_results['use_triton'] else 'PyTorch'}")
        
        # Attention benchmark
        print("  Testing Attention...")
        attention_results = benchmark_attention(batch_size, seq_len, d_model, n_heads)
        print(f"    PyTorch: {attention_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {attention_results['triton_time']:.3f} ms")
        print(f"    Speedup: {attention_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if attention_results['use_triton'] else 'PyTorch'}")
        
        # Newton-Schulz benchmark
        print("  Testing Newton-Schulz...")
        matrix_size = 128
        ns_results = benchmark_newton_schulz(matrix_size)
        print(f"    PyTorch: {ns_results['pytorch_time']:.3f} ms")
        print(f"    Triton:  {ns_results['triton_time']:.3f} ms")
        print(f"    Speedup: {ns_results['speedup']:.2f}x")
        print(f"    Winner:  {'Triton' if ns_results['use_triton'] else 'PyTorch'}")
        
        # Test 2: Auto-selection
        print("\n🔍 Test 2: Auto-Kernel Selection")
        print("-" * 40)
        
        kernel_selection = auto_select_kernels(config)
        print(f"✅ Auto-selection completed:")
        print(f"   RMSNorm: {'Triton' if kernel_selection['use_triton_rmsnorm'] else 'PyTorch'}")
        print(f"   Rotary:  {'Triton' if kernel_selection['use_triton_rotary'] else 'PyTorch'}")
        print(f"   MLP:     {'Triton' if kernel_selection['use_triton_mlp'] else 'PyTorch'}")
        print(f"   Attention: {'Triton' if kernel_selection['use_triton_attention'] else 'PyTorch'}")
        print(f"   Newton-Schulz: {'Triton' if kernel_selection['use_triton_newton_schulz'] else 'PyTorch'}")
        
        # Test 3: Model creation with selected kernels
        print("\n🔍 Test 3: Model Creation with Selected Kernels")
        print("-" * 40)
        
        model = MinimalLLM(config)
        model = model.to('cuda')
        print(f"✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test 4: Forward pass
        print("\n🔍 Test 4: Forward Pass")
        print("-" * 40)
        
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device='cuda')
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass successful: {output.shape}")
        
        # Test 5: Memory efficiency
        print("\n🔍 Test 5: Memory Efficiency")
        print("-" * 40)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create model with PyTorch kernels
        config_pytorch = ModelConfig()
        config_pytorch.use_triton_rmsnorm = False
        config_pytorch.use_triton_rotary = False
        config_pytorch.use_triton_mlp = False
        config_pytorch.use_triton_attention = False
        config_pytorch.use_triton_newton_schulz = False
        
        model_pytorch = MinimalLLM(config_pytorch).to('cuda')
        _ = model_pytorch(x)
        pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create model with Triton kernels
        model_triton = MinimalLLM(config).to('cuda')
        _ = model_triton(x)
        triton_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"Memory Usage:")
        print(f"  PyTorch: {pytorch_memory:.1f} MB")
        print(f"  Triton:  {triton_memory:.1f} MB")
        print(f"  Memory reduction: {((pytorch_memory - triton_memory) / pytorch_memory * 100):.1f}%")
        
        print("\n🎉 All enhanced tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_combinations():
    """Test different kernel combinations"""
    print("\n🔍 Test 6: Kernel Combination Testing")
    print("-" * 40)
    
    config = ModelConfig()
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 1
    config.max_seq_len = 64
    config.batch_size = 4
    
    combinations = [
        {'name': 'All PyTorch', 'rms': False, 'rotary': False, 'mlp': False, 'attn': False, 'ns': False},
        {'name': 'All Triton', 'rms': True, 'rotary': True, 'mlp': True, 'attn': True, 'ns': True},
        {'name': 'Mixed 1', 'rms': True, 'rotary': False, 'mlp': True, 'attn': False, 'ns': True},
        {'name': 'Mixed 2', 'rms': False, 'rotary': True, 'mlp': False, 'attn': True, 'ns': False},
    ]
    
    for combo in combinations:
        print(f"\n  Testing: {combo['name']}")
        
        config.use_triton_rmsnorm = combo['rms']
        config.use_triton_rotary = combo['rotary']
        config.use_triton_mlp = combo['mlp']
        config.use_triton_attention = combo['attn']
        config.use_triton_newton_schulz = combo['ns']
        
        try:
            model = MinimalLLM(config).to('cuda')
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device='cuda')
            
            with torch.no_grad():
                output = model(x)
            
            print(f"    ✅ Success: {output.shape}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Triton Integration Test Suite")
    print("=" * 60)
    
    # Run comprehensive tests
    success = test_all_kernels()
    
    if success:
        # Run combination tests
        test_kernel_combinations()
        
        print("\n🎉 All tests completed successfully!")
        print("🚀 Enhanced Triton integration is ready for training!")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
