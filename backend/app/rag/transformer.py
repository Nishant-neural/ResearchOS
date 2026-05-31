import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


MODEL_NAME = "google/flan-t5-base"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

def generate_text(prompt: str):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


def encode_text(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    encoder = model.get_encoder()

    outputs = encoder(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )

    return outputs

def extract_hidden_states(
    text: str,
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    encoder = model.get_encoder()

    with torch.no_grad():
        outputs = encoder(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    return {
        "last_hidden_state": outputs.last_hidden_state,
        "attention_mask": inputs["attention_mask"],
        "hidden_states": outputs.hidden_states,
    }
