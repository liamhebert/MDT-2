from typing import Callable
import torch
import transformers
from transformers import (
    Gemma3TextConfig,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
    Gemma3RotaryEmbedding,
)


def _bidirectional_window_overlay(
    sliding_window: int,
) -> Callable[[int, int, int, int], bool]:
    """
    Enables a bidirectional mask within the sliding window.
    """

    def inner_mask(
        batch_idx: int, head_idx: int, q_idx: int, kv_idx: int
    ) -> bool:
        """
        A token can attend to any other token if their absolute distance is
        within the (exclusive) sliding window size (distance < sliding_window).
        """
        return abs(q_idx - kv_idx) < sliding_window

    return inner_mask


def prepare_input(
    encoder_config: Gemma3TextConfig,
    encoder_embed_tokens: Gemma3TextScaledWordEmbedding,
    encoder_rotary_emb: Gemma3RotaryEmbedding,
    encoder_rotary_emb_local: Gemma3RotaryEmbedding,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    input_embeds = encoder_embed_tokens(input_ids)
    cache_position = torch.arange(
        0, input_ids.shape[1], device=input_ids.device
    )
    position_ids = cache_position.unsqueeze(0)

    graph_token_position = (attention_mask.sum(dim=1) - 2).clamp(min=0)

    mask_kwargs = {
        "config": encoder_config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    sliding_mask_kwargs = mask_kwargs.copy()

    mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
        True, dtype=torch.bool
    )
    sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
        encoder_config.sliding_window
    )

    causal_mask_mapping = {
        "full_attention": transformers.masking_utils.create_causal_mask(
            **mask_kwargs
        ),
        "sliding_attention": (
            transformers.masking_utils.create_sliding_window_causal_mask(
                **sliding_mask_kwargs
            )
        ),
    }

    # embed positions
    hidden_states = input_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings_global = encoder_rotary_emb(hidden_states, position_ids)
    position_embeddings_local = encoder_rotary_emb_local(
        hidden_states, position_ids
    )

    return (
        {
            "hidden_states": hidden_states,
            "position_embeddings_global": position_embeddings_global,
            "position_embeddings_local": position_embeddings_local,
            "causal_mask_mapping": causal_mask_mapping,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": False,
            "cache_position": cache_position,
        },
        graph_token_position,
        attention_mask,
    )
