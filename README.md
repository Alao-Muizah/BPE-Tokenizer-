# Byte Pair Encoding (BPE) Tokenizer From Scratch

A custom implementation of a Byte Pair Encoding (BPE) tokenizer built entirely from scratch in Python.
This project demonstrates how modern NLP tokenizers (used in LLMs) work under the hood.

# Features
Converts text into byte-level tokens
Learns merge rules using Byte Pair Encoding (BPE)
Builds a vocabulary dynamically during training
Encodes text into token IDs
Decodes token IDs back into text
Fully implemented without external NLP libraries

# How It Works

The tokenizer follows the Byte Pair Encoding (BPE) algorithm:
Convert text into ASCII byte tokens
Count frequency of adjacent token pairs
Merge the most frequent pair
Assign a new token ID to the merged pair
Repeat for a fixed number of merges
Build a vocabulary from learned merges

# Usage
1. Initialize Tokenizer
tokenizer = Tokenizer.Tokenize()
2. Train the Tokenizer
tokenizer.Training(text="your training corpus here", num_merges=20)
3. Encode Text
encoded = tokenizer.encode("language processing is fun")
print(encoded)
4. Decode Tokens
decoded = tokenizer.decode(encoded)
print(decoded)

# Purpose

Built as a learning project to understand tokenization and NLP fundamentals.
