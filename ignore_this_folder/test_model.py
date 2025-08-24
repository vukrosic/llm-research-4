#!/usr/bin/env python3
"""
Simple script to test the loaded model
"""

import torch
import torch.nn.functional as F
from ignore_this_folder.llm import MinimalLLM, ModelConfig
from data_server import CentralDataServer
import argparse
import os
import glob

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

def test_model_basic(model, device):
    """Test basic model functionality"""
    print(f"🧪 Testing basic model functionality...")
    
    # Test 1: Simple forward pass
    try:
        test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        print(f"   Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ Basic forward pass successful")
    except Exception as e:
        print(f"   ❌ Basic forward pass failed: {e}")
        return False
    
    # Test 2: Different input lengths
    try:
        test_input = torch.tensor([[1, 2, 3]], device=device)
        print(f"   Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ Variable length input successful")
    except Exception as e:
        print(f"   ❌ Variable length input failed: {e}")
        return False
    
    # Test 3: Batch processing
    try:
        test_input = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        print(f"   Input shape: {test_input.shape}")
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   ✅ Batch processing successful")
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
    
    return True

def test_model_generation(model, device, vocab_size):
    """Test text generation functionality"""
    print(f"\n🧪 Testing text generation functionality...")
    
    # Test 1: Simple token generation
    try:
        # Create a simple input sequence
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        print(f"   Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]  # Get last token logits
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            print(f"   Output shape: {outputs.shape}")
            print(f"   Next token logits shape: {next_token_logits.shape}")
            print(f"   Next token: {next_token.item()}")
            print(f"   ✅ Token generation successful")
    except Exception as e:
        print(f"   ❌ Token generation failed: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test loaded model functionality")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🔍 Using device: {device}")
    
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
            return
        checkpoint_path = latest_file
        print(f"📁 Using latest checkpoint: {checkpoint_path}")
    
    # Load data server to get tokenizer and vocab size
    print("📚 Loading tokenizer...")
    server = CentralDataServer()
    data = server.load_and_cache_data()
    vocab_size = data['vocab_size']
    
    # Load model
    print(f"🔄 Loading model from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        model_state_dict = checkpoint['model_state_dict']
        
        # Create model
        model = MinimalLLM(config)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Step: {checkpoint['step']}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Loss: {checkpoint['loss']:.4f}")
        print(f"   Vocabulary size: {vocab_size}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test model
    print(f"\n📊 Model Information:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   Vocabulary size: {config.vocab_size}")
    print(f"   Max sequence length: {config.max_seq_len}")
    
    # Run tests
    basic_success = test_model_basic(model, device)
    generation_success = test_model_generation(model, device, vocab_size)
    
    # Summary
    print(f"\n📋 Test Summary:")
    print(f"   Basic functionality: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   Generation functionality: {'✅ PASS' if generation_success else '❌ FAIL'}")
    
    if basic_success and generation_success:
        print(f"\n🎉 All tests passed! The model is working correctly.")
        print(f"💡 You can now use: python interactive_inference.py")
    else:
        print(f"\n⚠️ Some tests failed. Check the error messages above.")
        print(f"💡 The model may have compatibility issues.")

if __name__ == "__main__":
    main()
