#!/usr/bin/env python3
"""
Comprehensive benchmarking script for all Triton kernels
"""

import torch
import time
import numpy as np
from llm import (
    benchmark_rms_norm, benchmark_rotary, benchmark_mlp, 
    benchmark_attention, benchmark_newton_schulz
)

def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different configurations"""
    print("🚀 Comprehensive Triton vs PyTorch Performance Benchmark")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {'name': 'Small Model', 'batch_size': 16, 'seq_len': 256, 'd_model': 384, 'n_heads': 8, 'd_ff': 1536},
        {'name': 'Medium Model', 'batch_size': 32, 'seq_len': 512, 'd_model': 768, 'n_heads': 12, 'd_ff': 3072},
        {'name': 'Large Model', 'batch_size': 64, 'seq_len': 1024, 'd_model': 1024, 'n_heads': 16, 'd_ff': 4096},
        {'name': 'XL Model', 'batch_size': 128, 'seq_len': 2048, 'd_model': 1536, 'n_heads': 24, 'd_ff': 6144},
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n🔍 Configuration {i+1}: {config['name']}")
        print(f"   {config['batch_size']}x{config['seq_len']}x{config['d_model']} (heads: {config['n_heads']}, ff: {config['d_ff']})")
        print("-" * 60)
        
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        d_model = config['d_model']
        n_heads = config['n_heads']
        d_head = d_model // n_heads
        d_ff = config['d_ff']
        
        config_results = {}
        
        try:
            # RMSNorm benchmark
            print("  📊 RMSNorm:")
            rms_results = benchmark_rms_norm(batch_size, seq_len, d_model)
            config_results['rms'] = rms_results
            print(f"    PyTorch: {rms_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {rms_results['triton_time']:.3f} ms")
            print(f"    Speedup: {rms_results['speedup']:.2f}x")
            print(f"    Winner:  {'Triton' if rms_results['use_triton'] else 'PyTorch'}")
            
            # Rotary benchmark
            print("  📊 Rotary:")
            rotary_results = benchmark_rotary(batch_size, seq_len, n_heads, d_head)
            config_results['rotary'] = rotary_results
            print(f"    PyTorch: {rotary_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {rotary_results['triton_time']:.3f} ms")
            print(f"    Speedup: {rotary_results['speedup']:.2f}x")
            print(f"    Winner:  {'Triton' if rotary_results['use_triton'] else 'PyTorch'}")
            
            # MLP benchmark
            print("  📊 MLP:")
            mlp_results = benchmark_mlp(batch_size, seq_len, d_model, d_ff)
            config_results['mlp'] = mlp_results
            print(f"    PyTorch: {mlp_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {mlp_results['triton_time']:.3f} ms")
            print(f"    Speedup: {mlp_results['speedup']:.2f}x")
            print(f"    Winner:  {'Triton' if mlp_results['use_triton'] else 'PyTorch'}")
            
            # Attention benchmark
            print("  📊 Attention:")
            attention_results = benchmark_attention(batch_size, seq_len, d_model, n_heads)
            config_results['attention'] = attention_results
            print(f"    PyTorch: {attention_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {attention_results['triton_time']:.3f} ms")
            print(f"    Speedup: {attention_results['speedup']:.2f}x")
            print(f"    Winner:  {'Triton' if attention_results['use_triton'] else 'PyTorch'}")
            
            # Newton-Schulz benchmark
            print("  📊 Newton-Schulz:")
            matrix_size = min(256, d_model)
            ns_results = benchmark_newton_schulz(matrix_size)
            config_results['newton_schulz'] = ns_results
            print(f"    PyTorch: {ns_results['pytorch_time']:.3f} ms")
            print(f"    Triton:  {ns_results['triton_time']:.3f} ms")
            print(f"    Speedup: {ns_results['speedup']:.2f}x")
            print(f"    Winner:  {'Triton' if ns_results['use_triton'] else 'PyTorch'}")
            
            results[config['name']] = config_results
            
        except Exception as e:
            print(f"  ❌ Error in configuration {i+1}: {e}")
            continue
    
    return results

def analyze_results(results):
    """Analyze and summarize benchmark results"""
    print(f"\n📈 BENCHMARK ANALYSIS SUMMARY")
    print("=" * 70)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Calculate overall statistics
    all_speedups = []
    triton_wins = 0
    total_tests = 0
    
    for config_name, config_results in results.items():
        print(f"\n🔍 {config_name}:")
        
        for kernel_name, kernel_results in config_results.items():
            total_tests += 1
            speedup = kernel_results['speedup']
            all_speedups.append(speedup)
            
            if kernel_results['use_triton']:
                triton_wins += 1
                status = "✅ Triton"
            else:
                status = "❌ PyTorch"
            
            print(f"  {kernel_name:15}: {speedup:6.2f}x {status}")
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 40)
    print(f"  Total tests: {total_tests}")
    print(f"  Triton wins: {triton_wins} ({triton_wins/total_tests*100:.1f}%)")
    print(f"  PyTorch wins: {total_tests - triton_wins} ({(total_tests - triton_wins)/total_tests*100:.1f}%)")
    
    if all_speedups:
        print(f"  Average speedup: {np.mean(all_speedups):.2f}x")
        print(f"  Median speedup: {np.median(all_speedups):.2f}x")
        print(f"  Best speedup: {np.max(all_speedups):.2f}x")
        print(f"  Worst speedup: {np.min(all_speedups):.2f}x")
    
    # Performance recommendations
    print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
    print("-" * 40)
    
    if triton_wins > total_tests * 0.7:
        print("  🚀 Triton kernels show significant performance benefits!")
        print("  💡 Consider using Triton for production training")
    elif triton_wins > total_tests * 0.5:
        print("  ⚖️  Triton kernels show mixed performance benefits")
        print("  💡 Use auto-selection to get the best of both worlds")
    else:
        print("  ⚠️  PyTorch kernels perform better in most cases")
        print("  💡 Consider using PyTorch implementations or check hardware compatibility")

def memory_efficiency_test():
    """Test memory efficiency across different configurations"""
    print(f"\n💾 MEMORY EFFICIENCY TEST")
    print("=" * 50)
    
    configs = [
        {'batch_size': 32, 'seq_len': 512, 'd_model': 768, 'n_heads': 12, 'd_ff': 3072},
        {'batch_size': 64, 'seq_len': 1024, 'd_model': 1024, 'n_heads': 16, 'd_ff': 4096},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n  Configuration {i+1}: {config['batch_size']}x{config['seq_len']}x{config['d_model']}")
        print("-" * 50)
        
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        d_model = config['d_model']
        
        try:
            # Test RMSNorm memory usage
            x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.float16)
            
            # PyTorch RMSNorm
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            pytorch_norm = torch.nn.RMSNorm(d_model).to('cuda', dtype=torch.float16)
            _ = pytorch_norm(x)
            pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            # Triton RMSNorm
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            from llm import TritonRMSNormLayer
            triton_norm = TritonRMSNormLayer(d_model).to('cuda', dtype=torch.float16)
            _ = triton_norm(x)
            triton_memory = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"  RMSNorm Memory:")
            print(f"    PyTorch: {pytorch_memory:.1f} MB")
            print(f"    Triton:  {triton_memory:.1f} MB")
            print(f"    Reduction: {((pytorch_memory - triton_memory) / pytorch_memory * 100):.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def hardware_compatibility_test():
    """Test hardware compatibility and limitations"""
    print(f"\n🔧 HARDWARE COMPATIBILITY TEST")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {device_props.name}")
    print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"  Memory: {device_props.total_memory / 1024**3:.1f} GB")
    print(f"  Multiprocessors: {device_props.multi_processor_count}")
    
    # Check Triton availability
    try:
        import triton
        print(f"  Triton: ✅ Available (version: {triton.__version__})")
        
        # Test basic Triton functionality
        import triton.language as tl
        
        @triton.jit
        def test_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = x * 2.0
            tl.store(y_ptr + offsets, y, mask=mask)
        
        # Test kernel compilation
        x = torch.randn(1024, device='cuda')
        y = torch.empty_like(x)
        test_kernel[(1,)](x, y, 1024, 1024)
        print(f"  Triton Kernel Test: ✅ Passed")
        
    except ImportError:
        print(f"  Triton: ❌ Not available")
    except Exception as e:
        print(f"  Triton Kernel Test: ❌ Failed - {e}")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Triton Benchmark Suite")
    print("=" * 70)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires a GPU.")
        exit(1)
    
    print(f"🔧 CUDA Device: {torch.cuda.get_device_name()}")
    print(f"🔧 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark()
        
        # Analyze results
        analyze_results(results)
        
        # Memory efficiency test
        memory_efficiency_test()
        
        # Hardware compatibility test
        hardware_compatibility_test()
        
        print(f"\n✅ Comprehensive benchmark completed!")
        print(f"📊 Check the results above for performance insights")
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
