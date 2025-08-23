import torch
import torch.nn.functional as F
import os
import glob
from llm import MinimalLLM, ModelConfig, TextTokenDataset
from data_server import CentralDataServer
import argparse

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
    print(f"📁 Found latest checkpoint: {latest_file} (step {latest_step})")
    return latest_file

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"🔄 Loading model from {checkpoint_path}")
    
    try:
        # Try loading with weights_only=False for PyTorch 2.6+ compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"⚠️ Standard loading failed: {e}")
        print("🔄 Trying with weights_only=False for PyTorch 2.6+ compatibility...")
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
    
    return model, config

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
    """Generate text using the model"""
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(next(model.parameters()).device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(generated)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

def interactive_mode(model, tokenizer, device):
    """Interactive text generation mode"""
    print("\n🎭 Interactive Text Generation Mode")
    print("Type 'quit' to exit, 'help' for options")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n💬 Enter your prompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif prompt.lower() == 'help':
                print("\n📖 Available commands:")
                print("  quit - Exit the program")
                print("  help - Show this help message")
                print("  clear - Clear the screen")
                print("\n📝 Generation parameters:")
                print("  max_length: 100 tokens")
                print("  temperature: 0.8 (creativity)")
                print("  top_k: 50 (diversity)")
                print("  top_p: 0.9 (nucleus sampling)")
                continue
            elif prompt.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif not prompt:
                continue
            
            print(f"\n🤖 Generating text...")
            print(f"📝 Prompt: {prompt}")
            
            # Generate text
            generated_text = generate_text(
                model, tokenizer, prompt, 
                max_length=100, temperature=0.8, top_k=50, top_p=0.9
            )
            
            print(f"\n✨ Generated text:")
            print(f"{generated_text}")
            print(f"\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Interactive inference with trained LLM")
    parser.add_argument("--checkpoint", type=str, help="Path to specific checkpoint file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    
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
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("❌ No checkpoints found! Please train the model first.")
            print("💡 Run: python distributed_train.py")
            return
    
    # Load data server to get tokenizer
    print("📚 Loading tokenizer...")
    server = CentralDataServer()
    data = server.load_and_cache_data()
    tokenizer = data['tokenizer']
    
    # Load model
    try:
        model, config = load_model_from_checkpoint(checkpoint_path, device)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("💡 This might be a PyTorch 2.6+ compatibility issue.")
        print("💡 Try updating the checkpoint loading code or use an older PyTorch version.")
        return
    
    # Print model info
    print(f"\n📊 Model Information:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   Vocabulary size: {config.vocab_size}")
    print(f"   Max sequence length: {config.max_seq_len}")
    
    # Start interactive mode
    interactive_mode(model, tokenizer, device)

if __name__ == "__main__":
    main()
