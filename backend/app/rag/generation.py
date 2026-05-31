import torch
from transformers.modeling_outputs import BaseModelOutput

from app.rag.transformer import tokenizer, model


def generate_from_hidden_states(
    query: str,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
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
    
    Returns:
        Generated text
    """
    query_inputs = tokenizer(
        f"question: {query} answer using the retrieved context:",
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    device = next(model.parameters()).device
    query_inputs = {
        key: value.to(device)
        for key, value in query_inputs.items()
    }
    encoder_hidden_states = encoder_hidden_states.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        query_outputs = model.get_encoder()(
            **query_inputs,
            return_dict=True,
        )
        combined_hidden_states = torch.cat(
            [query_outputs.last_hidden_state, encoder_hidden_states],
            dim=1,
        )
        combined_attention_mask = torch.cat(
            [query_inputs["attention_mask"], attention_mask],
            dim=1,
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
                last_hidden_state=combined_hidden_states,
            ),
            attention_mask=combined_attention_mask,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )


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
