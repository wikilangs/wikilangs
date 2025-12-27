"""LLM utilities for Wikilangs.

This module provides utilities for integrating Wikilangs tokenizers with 
language models, including embedding freezing and language token addition.
"""

import os
from typing import List, Optional
try:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, AddedToken
    import sentencepiece as spm
    TORCH_AVAILABLE = True
except ImportError:
    # Define dummy types for type hints when libraries aren't available
    class PreTrainedModel: pass
    class PreTrainedTokenizerFast: pass
    class AddedToken: pass
    torch = None
    TORCH_AVAILABLE = False

from .tokenizer import tokenizer as load_tokenizer


def setup_embedding_freezing(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, tokens_to_freeze: List[str]):
    """
    Attaches gradient hooks to the model's embedding layers to freeze specific tokens during training.

    Args:
        model (PreTrainedModel): The model to modify.
        tokenizer (PreTrainedTokenizerFast): The model's tokenizer.
        tokens_to_freeze (list[str]): A list of token strings whose embeddings should be frozen.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and transformers are required for LLM utilities. "
                         "Install with: pip install torch transformers")
    
    if not tokens_to_freeze:
        return

    # Convert token strings to token IDs
    frozen_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_to_freeze), device=model.device)

    # Get the embedding and lm_head layers
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight

    def gradient_mask_hook(grad):
        """A hook to zero out gradients for frozen token IDs."""
        out = grad.clone()
        out[frozen_ids] = 0
        return out

    # Attach the hook to both embedding layers
    input_embeddings.register_hook(gradient_mask_hook)
    output_embeddings.register_hook(gradient_mask_hook)

    print(f"✅ Attached gradient hooks. Froze {len(frozen_ids)} tokens from being trained.")


def add_language_tokens(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerFast, 
    lang: str, 
    date: str = "latest",
    size: int = 16000,
    freeze_non_native_tokens: bool = False
):
    """
    Adds unique tokens from a Wikilangs tokenizer and optionally freezes base model tokens 
    that are not in the new custom tokenizer.
    
    Args:
        model (PreTrainedModel): The model to modify.
        tokenizer (PreTrainedTokenizerFast): The model's tokenizer to extend.
        lang (str): Language code (e.g., 'en', 'fr', 'ary').
        date (str): Date of the Wikilangs model (format: YYYYMMDD, default: "latest").
        size (int): Vocabulary size for the Wikilangs tokenizer (default: 16000).
        freeze_non_native_tokens (bool): Whether to freeze tokens not in the custom tokenizer.
        
    Returns:
        tuple: (modified_model, modified_tokenizer)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and transformers are required for LLM utilities. "
                         "Install with: pip install torch transformers")

    print(f"🌍 Loading Wikilangs tokenizer for lang='{lang}' with size={size}...")
    
    try:
        # Load the custom tokenizer using our wikilangs tokenizer loader
        custom_tokenizer = load_tokenizer(
            lang=lang, 
            date=date, 
            vocab_size=size, 
            format="sentencepiece"  # We need the SentencePiece format to extract vocab
        )
        
        # Get vocabularies for comparison
        base_vocab_set = set(tokenizer.get_vocab().keys())
        custom_vocab = custom_tokenizer.get_vocab()
        custom_vocab_set = set(custom_vocab.keys())
        
        print(f"📊 Base tokenizer vocab size: {len(base_vocab_set)}")
        print(f"📊 Custom tokenizer vocab size: {len(custom_vocab_set)}")
        
    except Exception as e:
        print(f"❌ Failed to load Wikilangs tokenizer. Error: {e}")
        return model, tokenizer

    # --- Identify tokens to add ---
    tokens_to_add = list(custom_vocab_set - base_vocab_set)
    if tokens_to_add:
        original_vocab_size = len(tokenizer)
        tokenizer.add_tokens([AddedToken(token, lstrip=True, rstrip=False) for token in tokens_to_add])
        model.resize_token_embeddings(len(tokenizer))
        print(f"✅ Added {len(tokens_to_add)} unique tokens. New vocab size: {len(tokenizer)}")
    else:
        print("✅ No new unique tokens found to add.")
        
    # --- Freeze non-native tokens if requested ---
    if freeze_non_native_tokens:
        tokens_to_freeze = list(base_vocab_set - custom_vocab_set)
        print(f"🧊 Freezing {len(tokens_to_freeze)} non-native tokens...")
        # We pass the original model and tokenizer before resizing to the freeze helper
        setup_embedding_freezing(model, tokenizer, tokens_to_freeze)

    return model, tokenizer


def load_wikilangs_tokenizer_for_llm(lang: str, date: str = "latest", vocab_size: int = 16000):
    """
    Load a Wikilangs tokenizer in HuggingFace format for LLM integration.
    
    Args:
        lang (str): Language code (e.g., 'en', 'fr', 'ary')
        date (str): Date of the model (format: YYYYMMDD, default: "latest")
        vocab_size (int): Vocabulary size (default: 16000)
        
    Returns:
        PreTrainedTokenizerFast: HuggingFace-compatible tokenizer
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch and transformers are required for LLM utilities. "
                         "Install with: pip install torch transformers")
    
    # Load tokenizer in HuggingFace format
    wikilangs_tokenizer = load_tokenizer(
        lang=lang,
        date=date,
        vocab_size=vocab_size,
        format="huggingface"
    )
    
    return wikilangs_tokenizer.tokenizer
