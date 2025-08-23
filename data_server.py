import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pickle
import os
from tqdm import tqdm

class CentralDataServer:
    def __init__(self, num_documents=500, max_tokens=200000, max_seq_len=512):
        self.num_documents = num_documents
        self.max_tokens = max_tokens
        self.max_seq_len = max_seq_len
        self.cache_dir = "data_cache"
        
    def load_and_cache_data(self):
        """Load and cache tokenized data centrally"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = f"{self.cache_dir}/thebluescrubs_tokenized_data_{self.num_documents}_{self.max_tokens}.pkl"

        if os.path.exists(cache_file):
            print(f"📦 Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data

        print(f"🔄 Processing new data (will cache for future use)")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        dataset = load_dataset("openmed-community/TheBlueScrubs-v1-fixed", split="train", streaming=True, token=False)
        
        texts = []
        for i, item in enumerate(dataset):
            if i >= self.num_documents:
                break
            text = item.get("text", "")
            if text:
                texts.append(text[:3000])

        # Tokenize
        print("Tokenizing texts...")
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        tokens = all_tokens[:self.max_tokens]
        
        # Cache the processed data
        cached_data = {
            'texts': texts, 
            'tokenizer': tokenizer, 
            'tokens': tokens,
            'vocab_size': tokenizer.vocab_size
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)

        print(f"💾 Cached data to {cache_file}")
        return cached_data

if __name__ == "__main__":
    # Test data loading
    server = CentralDataServer()
    data = server.load_and_cache_data()
    print(f"✅ Loaded {len(data['texts'])} documents, {len(data['tokens']):,} tokens")
