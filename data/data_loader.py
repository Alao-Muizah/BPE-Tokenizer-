from datasets import load_dataset


# -------------------------
# DATA LOADERS
# -------------------------

def load_wikitext():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return "\n".join(dataset["train"]["text"])


def load_tinystories():
    dataset = load_dataset("roneneldan/TinyStories")
    return "\n".join(dataset["train"]["text"])