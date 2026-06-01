import torch
from transformers.modeling_outputs import BaseModelOutput

from app.rag.transformer import tokenizer, model


def generate_from_hidden_states(
    query: str,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    conditioning_strategy: str = "framed_memory",
) -> str:
    """
    Generate response using pre-computed encoder hidden states.
    
    Skips the encoder entirely:
    - No tokenization of retrieved text
    - No encoder forward pass
    - Direct decoder usage with pre-computed hidden states
    
    Args:
        query: The user query (used for context)
        encoder_hidden_states: Pre-computed document states [batch, seq_len, hidden_dim]
        attention_mask: Attention mask for the pre-computed document states.
        conditioning_strategy: How to frame cached memory for the decoder.
    
    Returns:
        Generated text
    """
    device = next(model.parameters()).device
    encoder_hidden_states = encoder_hidden_states.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        conditioning = _build_conditioning_memory(
            query=query,
            memory_hidden_states=encoder_hidden_states,
            memory_attention_mask=attention_mask,
            strategy=conditioning_strategy,
            device=device,
        )
        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = model.config.pad_token_id
        decoder_input_ids = torch.tensor(
            [[decoder_start_token_id]],
            dtype=torch.long,
            device=device,
        )

        outputs = model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=BaseModelOutput(
                last_hidden_state=conditioning["hidden_states"],
            ),
            attention_mask=conditioning["attention_mask"],
            max_new_tokens=96,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


def _build_conditioning_memory(
    query: str,
    memory_hidden_states: torch.Tensor,
    memory_attention_mask: torch.Tensor,
    strategy: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if strategy == "query_prefix":
        prefix = (
            "Answer the question directly using only the retrieved latent memory. "
            "Do not summarize the whole document. "
            "If the answer is not present, say unknown.\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        prefix_memory = _encode_conditioning_text(prefix, device)
        return _concat_memory_parts(
            [
                prefix_memory["hidden_states"],
                memory_hidden_states,
            ],
            [
                prefix_memory["attention_mask"],
                memory_attention_mask,
            ],
        )

    if strategy == "framed_memory":
        prefix = (
            "Task: answer the user's question using the relevant latent memory. "
            "The following hidden states are retrieved context, not the answer. "
            "Use them only as evidence.\n\n"
            f"Question: {query}\n"
            "Relevant latent memory begins:"
        )
        suffix = (
            "Relevant latent memory ended.\n\n"
            f"Question: {query}\n"
            "Give a direct answer. Do not restate the whole thesis. "
            "If the retrieved memory does not contain the answer, say unknown.\n"
            "Answer:"
        )
        prefix_memory = _encode_conditioning_text(prefix, device)
        suffix_memory = _encode_conditioning_text(suffix, device)
        return _concat_memory_parts(
            [
                prefix_memory["hidden_states"],
                memory_hidden_states,
                suffix_memory["hidden_states"],
            ],
            [
                prefix_memory["attention_mask"],
                memory_attention_mask,
                suffix_memory["attention_mask"],
            ],
        )

    raise ValueError(
        f"Unknown conditioning strategy {strategy!r}. "
        "Use 'framed_memory' or 'query_prefix'."
    )


def _encode_conditioning_text(
    text: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }
    outputs = model.get_encoder()(
        **inputs,
        return_dict=True,
    )
    return {
        "hidden_states": outputs.last_hidden_state,
        "attention_mask": inputs["attention_mask"],
    }


def _concat_memory_parts(
    hidden_state_parts: list[torch.Tensor],
    attention_mask_parts: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.cat(hidden_state_parts, dim=1),
        "attention_mask": torch.cat(attention_mask_parts, dim=1),
    }


def generate_from_text_with_encoding(
    prompt: str,
) -> str:
    """
    Baseline generation: encode text and generate.
    Used for comparison against hidden-state approach.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )
