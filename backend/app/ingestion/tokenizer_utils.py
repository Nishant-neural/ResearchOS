from transformers import AutoTokenizer

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text: str) -> int:
    tokens = tokenizer.encode(text)
    return len(tokens)


def tokenize_text(text: str):
    return tokenizer.encode(text)


def decode_tokens(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)