#!/usr/bin/env python3
"""
Tokenization Analysis Script
Analyzes how well different tokenizers handle medical text from TheBlueScrubs dataset
"""

import os
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import re

def load_sample_data(num_docs=50):
    """Load a sample of medical texts for analysis"""
    print("📚 Loading sample medical texts...")
    
    dataset = load_dataset("openmed-community/TheBlueScrubs-v1-fixed", split="train", streaming=True, token=False)
    
    texts = []
    for i, item in enumerate(dataset):
        if i >= num_docs:
            break
        text = item.get("text", "")
        if text:
            texts.append(text[:2000])  # First 2000 chars for analysis
    
    print(f"✅ Loaded {len(texts)} sample documents")
    return texts

def analyze_tokenizer(tokenizer_name, texts):
    """Analyze how well a tokenizer handles the medical texts"""
    print(f"\n🔍 Analyzing tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Failed to load {tokenizer_name}: {e}")
        return None
    
    # Analyze tokenization
    total_chars = 0
    total_tokens = 0
    token_lengths = []
    compression_ratios = []
    unknown_tokens = 0
    medical_terms = []
    
    # Sample medical terms to check
    medical_terms_to_check = [
        "diagnosis", "symptoms", "treatment", "medication", "patient", "doctor",
        "hospital", "emergency", "surgery", "prescription", "antibiotics",
        "cardiovascular", "neurological", "respiratory", "gastrointestinal",
        "hypertension", "diabetes", "cancer", "infection", "inflammation"
    ]
    
    for text in tqdm(texts, desc=f"Tokenizing with {tokenizer_name}"):
        # Basic tokenization stats
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chars = len(text)
        
        total_chars += chars
        total_tokens += len(tokens)
        token_lengths.append(len(tokens))
        compression_ratios.append(chars / len(tokens))
        
        # Check for unknown tokens
        if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None:
            unknown_tokens += tokens.count(tokenizer.unk_token_id)
        
        # Check medical term tokenization
        for term in medical_terms_to_check:
            if term.lower() in text.lower():
                term_tokens = tokenizer.encode(term, add_special_tokens=False)
                medical_terms.append({
                    'term': term,
                    'tokens': term_tokens,
                    'token_count': len(term_tokens),
                    'decoded': tokenizer.decode(term_tokens)
                })
    
    # Calculate statistics
    avg_compression = np.mean(compression_ratios)
    avg_tokens_per_doc = np.mean(token_lengths)
    vocab_size = tokenizer.vocab_size
    
    # Analyze medical term tokenization
    medical_term_stats = {}
    for term_info in medical_terms:
        term = term_info['term']
        if term not in medical_term_stats:
            medical_term_stats[term] = []
        medical_term_stats[term].append(term_info['token_count'])
    
    # Calculate average tokens per medical term
    avg_tokens_per_medical_term = {}
    for term, counts in medical_term_stats.items():
        avg_tokens_per_medical_term[term] = np.mean(counts)
    
    return {
        'name': tokenizer_name,
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'avg_compression_ratio': avg_compression,
        'avg_tokens_per_doc': avg_tokens_per_doc,
        'vocab_size': vocab_size,
        'unknown_tokens': unknown_tokens,
        'unknown_rate': unknown_tokens / total_tokens if total_tokens > 0 else 0,
        'medical_terms': medical_terms,
        'avg_tokens_per_medical_term': avg_tokens_per_medical_term,
        'tokenizer': tokenizer
    }

def compare_tokenizers(tokenizer_names, texts):
    """Compare multiple tokenizers"""
    results = []
    
    for name in tokenizer_names:
        result = analyze_tokenizer(name, texts)
        if result:
            results.append(result)
    
    return results

def print_analysis_results(results):
    """Print detailed analysis results"""
    print("\n" + "="*80)
    print("�� TOKENIZER ANALYSIS RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n🔍 {result['name']}")
        print(f"   �� Compression ratio: {result['avg_compression_ratio']:.2f} chars/token")
        print(f"   📄 Avg tokens per document: {result['avg_tokens_per_doc']:.1f}")
        print(f"   �� Vocabulary size: {result['vocab_size']:,}")
        print(f"   ❓ Unknown tokens: {result['unknown_tokens']:,} ({result['unknown_rate']*100:.2f}%)")
        
        print(f"\n   �� Medical term tokenization:")
        for term, avg_tokens in result['avg_tokens_per_medical_term'].items():
            print(f"      {term}: {avg_tokens:.1f} tokens")
    
    # Find best tokenizer
    best_compression = max(results, key=lambda x: x['avg_compression_ratio'])
    best_unknown_rate = min(results, key=lambda x: x['unknown_rate'])
    
    print(f"\n🏆 RECOMMENDATIONS:")
    print(f"   Best compression: {best_compression['name']} ({best_compression['avg_compression_ratio']:.2f} chars/token)")
    print(f"   Best unknown rate: {best_unknown_rate['name']} ({best_unknown_rate['unknown_rate']*100:.2f}% unknown)")
    
    # Medical-specific recommendation
    medical_scores = []
    for result in results:
        # Score based on medical term efficiency and unknown rate
        medical_efficiency = np.mean(list(result['avg_tokens_per_medical_term'].values()))
        score = (1 / medical_efficiency) * (1 - result['unknown_rate'])
        medical_scores.append((result['name'], score))
    
    best_medical = max(medical_scores, key=lambda x: x[1])
    print(f"   Best for medical text: {best_medical[0]} (score: {best_medical[1]:.3f})")

def analyze_specific_medical_texts(results, texts):
    """Analyze specific medical text examples"""
    print(f"\n" + "="*80)
    print("🏥 DETAILED MEDICAL TEXT ANALYSIS")
    print("="*80)
    
    # Find a good medical text sample
    medical_keywords = ['diagnosis', 'patient', 'treatment', 'symptoms', 'medication']
    sample_text = None
    
    for text in texts:
        if any(keyword in text.lower() for keyword in medical_keywords):
            sample_text = text[:500]  # First 500 chars
            break
    
    if not sample_text:
        print("❌ No suitable medical text sample found")
        return
    
    print(f"\n📄 Sample medical text (first 500 chars):")
    print(f"   {sample_text}...")
    
    for result in results:
        print(f"\n🔍 Tokenization with {result['name']}:")
        tokens = result['tokenizer'].encode(sample_text, add_special_tokens=False)
        decoded = result['tokenizer'].decode(tokens[:50])  # First 50 tokens
        
        print(f"   Tokens: {len(tokens)}")
        print(f"   First 50 tokens decoded: {decoded}...")
        
        # Check for unknown tokens
        if hasattr(result['tokenizer'], 'unk_token_id') and result['tokenizer'].unk_token_id is not None:
            unknown_count = tokens.count(result['tokenizer'].unk_token_id)
            if unknown_count > 0:
                print(f"   ⚠️  Contains {unknown_count} unknown tokens")

def main():
    """Main analysis function"""
    print("�� Medical Text Tokenizer Analysis")
    print("="*50)
    
    # Load sample medical texts
    texts = load_sample_data(num_docs=100)
    
    # Define tokenizers to test
    tokenizer_names = [
        "HuggingFaceTB/SmolLM-135M",  # Current tokenizer
        "microsoft/DialoGPT-medium",   # General purpose
        "gpt2",                        # GPT-2 tokenizer
        "EleutherAI/gpt-neo-125M",    # GPT-Neo
        "facebook/opt-125m",          # OPT
        "microsoft/DialoGPT-small",   # Smaller alternative
    ]
    
    print(f"\n�� Testing {len(tokenizer_names)} tokenizers...")
    
    # Analyze all tokenizers
    results = compare_tokenizers(tokenizer_names, texts)
    
    # Print results
    print_analysis_results(results)
    
    # Detailed medical text analysis
    analyze_specific_medical_texts(results, texts)
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Check the compression ratio - higher is better (more chars per token)")
    print(f"   2. Lower unknown token rate is better")
    print(f"   3. Medical terms should be tokenized efficiently (fewer tokens per term)")
    print(f"   4. Consider vocabulary size vs. your model capacity")
    
    # Save results for later reference
    with open("tokenizer_analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to tokenizer_analysis_results.pkl")

if __name__ == "__main__":
    main()
