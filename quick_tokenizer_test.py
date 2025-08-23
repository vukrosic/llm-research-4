#!/usr/bin/env python3
"""
Quick test of the current SmolLM tokenizer on medical text
"""

from transformers import AutoTokenizer
from datasets import load_dataset

def test_current_tokenizer():
    """Test the current SmolLM tokenizer on medical text"""
    print("�� Testing SmolLM-135M tokenizer on medical text...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Loaded tokenizer: {tokenizer.name_or_path}")
    print(f"�� Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Load a few medical texts
    dataset = load_dataset("openmed-community/TheBlueScrubs-v1-fixed", split="train", streaming=True, token=False)
    
    medical_terms = [
        "diagnosis", "symptoms", "treatment", "medication", "patient", "doctor",
        "hospital", "emergency", "surgery", "prescription", "antibiotics",
        "cardiovascular", "neurological", "respiratory", "gastrointestinal",
        "hypertension", "diabetes", "cancer", "infection", "inflammation",
        "myocardial infarction", "cerebrovascular accident", "pneumonia"
    ]
    
    print(f"\n🏥 Testing medical term tokenization:")
    for term in medical_terms:
        tokens = tokenizer.encode(term, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        print(f"   {term:25} -> {len(tokens):2d} tokens: {decoded}")
    
    # Test on actual medical text
    print(f"\n📄 Testing on actual medical text:")
    for i, item in enumerate(dataset):
        if i >= 3:  # Just test first 3 documents
            break
        
        text = item.get("text", "")[:300]  # First 300 chars
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            compression_ratio = len(text) / len(tokens)
            
            print(f"\n   Document {i+1}:")
            print(f"   Text: {text}...")
            print(f"   Tokens: {len(tokens)}")
            print(f"   Compression: {compression_ratio:.2f} chars/token")
            
            # Check for unknown tokens
            if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
                unknown_count = tokens.count(tokenizer.unk_token_id)
                if unknown_count > 0:
                    print(f"   ⚠️  Contains {unknown_count} unknown tokens")
    
    print(f"\n💡 Analysis complete!")
    print(f"   - Higher compression ratio = better (more chars per token)")
    print(f"   - Fewer tokens per medical term = better")
    print(f"   - Lower unknown token rate = better")

if __name__ == "__main__":
    test_current_tokenizer()
