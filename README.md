#  BPE Tokenizer (From Scratch)

A Byte Pair Encoding (BPE) tokenizer built from scratch in Python, featuring:

* Pretrained tokenizer inference
* Custom tokenizer training
* Dataset + file-based training
* CLI interface
* Streamlit visualization studio
* Token breakdown visualization

---

# Features

###  Tokenizer Core

* Byte-level BPE implementation
* Trainable merge rules
* UTF-8 safe encoding/decoding
* Vocabulary reconstruction

###  Training Modes

* Train from HuggingFace datasets

  * WikiText
  * TinyStories
* Train from custom `.txt` files

###  Interfaces

* CLI tool (interactive + training)
* Streamlit web app
* Save/load tokenizer models

###  Visualization

* Token ID mapping
* Base vs merged token labeling
* Step-by-step token breakdown

---

# Installation

```bash

pip install -r requirements.txt
```

---

#  Project Structure

```
bpe-tokenizer/
│
├── tokenizer/
│   └── tokenizer.py
│
├── data/
│   └── data_loader.py
│
├── train.py
├── cli.py
├── app.py
└── saved/   # pretrained tokenizer
```

---

#  Training the Tokenizer

You can train from a specified file path.

---

```bash
python train.py --file data/corpus.txt --merges 500 --save_path saved
```

---

#  CLI Usage

Run interactive CLI:

```bash
python cli.py
```

---

##  Mode 1: Pretrained Tokenizer

* Load saved tokenizer
* Encode/decode text
* View token breakdown

Exit anytime:

```text
q
```

---

##  Mode 2: Train Tokenizer

You can train from:

* dataset (`wiki`, `stories`)
* custom file path

Example:

```text
Enter task (1 or 2): 2
Train from dataset (wiki/stories) or file path: data/corpus.txt
Save path: saved_user
Number of merges: 500
```

Exit anytime:

```text
q
```

---

#  Streamlit App

Run UI:

```bash
streamlit run app.py
```

---

##  Features

###  Pretrained Mode

* Encode text
* View token IDs
* Visual token breakdown
* Decoded output

### Training Mode

* Upload `.txt` dataset
* Choose number of merges
* Train tokenizer
* Download model as ZIP

### Custom Model Testing

* Test newly trained tokenizer in UI
* Real-time encoding/decoding

---

#  Tokenizer Design

### Encoding Pipeline

1. UTF-8 byte conversion
2. Pre-tokenization
3. BPE merge iterations
4. Merge rule application

### Decoding Pipeline

1. Reverse merge rules
2. Reconstruct bytes
3. Decode UTF-8 text

---

#  Token Output Format

Each token is stored as:

```python
{
    "token_id": 262,
    "token_text": "of "
}
```

Used for:

* visualization
* debugging
* UI display

---

#  Saving Model

Saved components:

* `merges.json` → BPE merge rules
* `vocab.json` → reconstructed vocabulary

---

#  Notes

* Always use UTF-8 text
* Delete old saved models after changing tokenizer logic
* Training on large datasets may take time (pure Python implementation)

---

#  Future Improvements

* GPU acceleration (optional)
* Faster merge algorithm (heap optimization)
* HuggingFace tokenizer compatibility
* Streaming dataset training
* Token frequency analytics dashboard

---

#  Author

Built from scratch as a learning project in:

* Tokenization
* NLP preprocessing
* Byte Pair Encoding (BPE)
* Streamlit + CLI tooling

