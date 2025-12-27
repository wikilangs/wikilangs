"""Wikilangs: A Python package for consuming Wikipedia language models and datasets.

This package provides easy access to pre-trained language models and datasets including:
- Tokenizers (BPE) with HuggingFace format support
- N-gram models (for scoring and next token prediction)
- Markov chains (for text generation)
- Vocabularies (using vocabulous)
- Datasets (Wikipedia text data in various splits)
- LLM utilities (embedding freezing, language token addition)

Each model and dataset can be initialized independently with parameters for date, language,
and model/dataset-specific settings.

Example:
    from wikilangs import tokenizer, ngram, markov, vocabulary, languages
    from wikilangs.llm import add_language_tokens, setup_embedding_freezing
    
    # Create a tokenizer (date defaults to 'latest')
    tok = tokenizer(lang='en', vocab_size=16000)
    
    # Create a tokenizer in HuggingFace format
    hf_tok = tokenizer(lang='en', vocab_size=16000, format='huggingface')
    
    # Create an n-gram model
    ng = ngram(lang='en', gram_size=3)
    
    # Create a markov chain
    mc = markov(lang='en', depth=2)
    
    # Create a vocabulary
    vocab = vocabulary(lang='en')
    
    # Create embeddings
    emb = embeddings(lang='ary', dimension=32)
    
    # Add language tokens to an LLM
    model, tokenizer = add_language_tokens(model, tokenizer, 'ary', vocab_size=16000)
"""

from importlib import metadata
__version__ = metadata.version("wikilangs")
__author__ = "Omar Kamali"

# Import factory functions to make them available at package level
from .tokenizer import tokenizer
from .ngram import ngram
from .markov import markov
from .vocabulary import vocabulary
from .embeddings import embeddings
from .languages import languages, languages_with_metadata, LanguageInfo, get_language_info

# Import LLM utilities (optional - only available if dependencies are installed)
try:
    from .llm import setup_embedding_freezing, add_language_tokens, load_wikilangs_tokenizer_for_llm
    _llm_functions = ["setup_embedding_freezing", "add_language_tokens", "load_wikilangs_tokenizer_for_llm"]
except ImportError:
    _llm_functions = []

__all__ = ["tokenizer", "ngram", "markov", "vocabulary", "embeddings", "languages", "languages_with_metadata", "LanguageInfo", "get_language_info"] + _llm_functions
