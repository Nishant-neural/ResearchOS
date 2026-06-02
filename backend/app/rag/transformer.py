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
        max_length=768,
    )
    # ADD HERE
    token_count = inputs["input_ids"].shape[1]

    print(f"[DEBUG] Prompt tokens: {token_count}")

    decoded_prompt = tokenizer.decode(
    inputs["input_ids"][0],
    skip_special_tokens=False,
)

    print("\n[DEBUG] FINAL PROMPT:")
    print(decoded_prompt[-3000:])
    print("\n=====================\n")
    
    outputs = model.generate(
        **inputs,
       max_new_tokens=128,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
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
