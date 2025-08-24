#!/usr/bin/env python3
"""
Simple test script to verify dtype handling is fixed
"""

import torch
import torch.nn as nn

def test_dtype_handling():
    """Test that all classes handle dtype conversion properly"""
    print("🧪 Testing Dtype Handling")
    print("=" * 40)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU operations.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    
    try:
        # Test 1: RMSNorm
        print("\n🔍 Test 1: RMSNorm Dtype Handling")
        from ignore_this_folder.llm import TritonRMSNormLayer
        
        rms_norm = TritonRMSNormLayer(128)
        rms_norm = rms_norm.to('cuda', dtype=torch.float16)
        
        x = torch.randn(4, 64, 128, device='cuda', dtype=torch.float16)
        output = rms_norm(x)
        print(f"✅ RMSNorm: Input {x.dtype}, Output {output.dtype}, Weight {rms_norm.weight.dtype}")
        
        # Test 2: Rotary
        print("\n🔍 Test 2: Rotary Dtype Handling")
        from ignore_this_folder.llm import Rotary
        
        rotary = Rotary(64, 128)
        rotary = rotary.to('cuda', dtype=torch.float16)
        
        q = torch.randn(4, 8, 64, 64, device='cuda', dtype=torch.float16)
        k = torch.randn(4, 8, 64, 64, device='cuda', dtype=torch.float16)
        
        q_out, k_out = rotary(q), rotary(k)
        print(f"✅ Rotary: Input {q.dtype}, Output {q_out.dtype}, Buffers {rotary.cos.dtype}")
        
        # Test 3: MultiHeadAttention
        print("\n🔍 Test 3: MultiHeadAttention Dtype Handling")
        from ignore_this_folder.llm import MultiHeadAttention
        
        attention = MultiHeadAttention(128, 8, 64)
        attention = attention.to('cuda', dtype=torch.float16)
        
        x = torch.randn(4, 64, 128, device='cuda', dtype=torch.float16)
        output = attention(x)
        print(f"✅ Attention: Input {x.dtype}, Output {output.dtype}, Weights {attention.qkv.weight.dtype}")
        
        # Test 4: FeedForward
        print("\n🔍 Test 4: FeedForward Dtype Handling")
        from ignore_this_folder.llm import FeedForward
        
        ff = FeedForward(128, 512)
        ff = ff.to('cuda', dtype=torch.float16)
        
        x = torch.randn(4, 64, 128, device='cuda', dtype=torch.float16)
        output = ff(x)
        print(f"✅ FeedForward: Input {x.dtype}, Output {output.dtype}, Weights {ff.linear1.weight.dtype}")
        
        print("\n🎉 All dtype tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dtype_handling()
    if success:
        print("\n🚀 Dtype handling is working correctly!")
    else:
        print("\n⚠️  Dtype handling has issues. Check the error messages above.")
