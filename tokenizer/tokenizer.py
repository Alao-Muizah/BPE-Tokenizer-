import json
import os


class Tokenizer:

    def __init__(self):
        self.merge_rules = []   
        self.vocab = {}
        self.next_id = 256
        self.is_trained = False

    # -------------------------
    # PRETOKENIZATION
    # -------------------------
    def pretokenize(self, text):
        tokens = []
        current = ""

        for c in text:

            if c == " ":
                if current:  
                    tokens.append(current)
                    current = ""
                tokens.append(" ")

            elif c in ",.!?;:":
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(c)

            else:
                current += c 

        if current:
            tokens.append(current)

        return tokens

    # -------------------------
    # TEXT → BYTES
    # -------------------------

    def _get_bytes(self, text):
        return list(text.encode("utf-8"))
    
    # -------------------------
    # PAIR COUNTING
    # -------------------------
    def _count_pairs(self, tokens):
        counts = {}

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    # -------------------------
    # BEST PAIR
    # -------------------------
    def _get_most_frequent_pair(self, pair_counts):
        return max(pair_counts.items(), key=lambda x: x[1])

    # -------------------------
    # MERGE PAIR
    # -------------------------
    def _merge_pair(self, tokens, pair, new_id):

        result = []
        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1 and
                tokens[i] == pair[0] and
                tokens[i + 1] == pair[1]
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    # -------------------------
    # TRAIN
    # -------------------------
    def train(self, text, num_merges=50): 

        chunks = self.pretokenize(text)

        tokens = []
        for chunk in chunks:
            tokens.extend(self._get_bytes(chunk))

        for i in range(num_merges):

            pair_counts = self._count_pairs(tokens)

            if not pair_counts:
                break

            best_pair, best_count = self._get_most_frequent_pair(pair_counts)

            if best_count < 2:
                break

            tokens = self._merge_pair(tokens, best_pair, self.next_id)

            self.merge_rules.append((best_pair, self.next_id))

            self.next_id += 1

            print(
            f"Merge {i+1}/{num_merges} | "
            f"Pair: {best_pair} → {self.next_id - 1}     | "
            f"Freq: {best_count}"
        )
            
        self._build_vocab()
        
        self.merge_rules_dict = {
            new_id: pair
            for pair, new_id in self.merge_rules
        }

        self.is_trained = True

    # -------------------------
    # BUILD VOCAB
    # -------------------------
    def _build_vocab(self):

        vocab = {i: bytes([i]) for i in range(256)} 

        for (a, b), new_id in self.merge_rules:
            vocab[new_id] = vocab[a] + vocab[b]

        self.vocab = vocab

    # -------------------------
    # ENCODE
    # -------------------------
    def encode(self, text):

        tokens = []
        
        for chunk in self.pretokenize(text):
            tokens.extend(self._get_bytes(chunk))

        if not self.merge_rules:
            return tokens

        while True:
            merged = False

            for pair, new_id in self.merge_rules:
                new_tokens = self._merge_pair(tokens, pair, new_id)

                if new_tokens != tokens:
                    tokens = new_tokens
                    merged = True

            if not merged:
                break

        token_info = []

        for t in tokens:

            token_text = self.decode_id(t).decode("utf-8", errors='ignore')
            token_info.append({
            "token_id": t,
            "token_text": token_text
        })

        return tokens, token_info
    
    # -------------------------
    # DECODE
    # -------------------------

    def decode_id(self, t):

        if t < 256:
            return bytes([t])

        a, b = self.merge_rules_dict[t]

        return self.decode_id(a) + self.decode_id(b)
    
    def decode(self, token_ids):

        out_bytes = b""

        for t in token_ids:
            out_bytes += self.decode_id(t)

        return out_bytes.decode("utf-8", errors="ignore")
 
    # -------------------------
    # SAVE MODEL
    # -------------------------
    def save(self, path):

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "merges.json"), "w") as f:
            json.dump([
                {"pair": list(pair), "id": id_}
                for pair, id_ in self.merge_rules
            ], f)

        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(
                {str(k): list(v) for k, v in self.vocab.items()},
                f
            )

    # -------------------------
    # LOAD MODEL
    # -------------------------
    def load(self, path):

        with open(os.path.join(path, "merges.json"), "r") as f:
            merges = json.load(f)

        self.merge_rules = []
        max_id = 255

        for item in merges:
            pair = tuple(item["pair"])
            new_id = item["id"]
            self.merge_rules.append((pair, new_id))
            max_id = max(max_id, new_id)

        self.next_id = max_id + 1

        self.merge_rules_dict = {
        new_id: pair
        for pair, new_id in self.merge_rules
        } 
        with open(os.path.join(path, "vocab.json"), "r") as f:
            vocab_json = json.load(f)

        self.vocab = {
            int(k): bytes(v)
            for k, v in vocab_json.items()
        }

        self.is_trained = True



