"""Tokenizer module for Wikilangs.

This module provides functionality to load and use BPE tokenizers
trained on Wikipedia data for different languages and dates.
"""

import os
from typing import Optional, Union, List, Literal
try:
    import sentencepiece as spm
except ImportError:
    raise ImportError("sentencepiece is required for tokenizer functionality. Install with: pip install sentencepiece")
from huggingface_hub import hf_hub_download

# Optional HuggingFace imports for format conversion
try:
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BPETokenizer:
    """BPE Tokenizer for Wikipedia data.
    
    This class loads pre-trained BPE tokenizers from HuggingFace
    based on date, language, and vocabulary size.
    """
    
    def __init__(self, lang: str, date: str = "latest", vocab_size: int = 16000, 
                 format: Literal["sentencepiece", "huggingface"] = "sentencepiece"):
        """Initialize the BPE tokenizer.
        
        Args:
            lang (str): Language code (e.g., 'en', 'fr', 'ary')
            date (str): Date of the model (format: YYYYMMDD, default: "latest")
            vocab_size (int): Vocabulary size (default: 16000)
            format (str): Output format - "sentencepiece" or "huggingface" (default: "sentencepiece")
        
        Raises:
            FileNotFoundError: If the model files cannot be found
            ValueError: If vocab_size is not supported or format is invalid
            ImportError: If huggingface format is requested but transformers is not available
        """
        self.date = date
        self.lang = lang
        self._vocab_size = vocab_size
        self.format = format
        
        # Validate format
        if format not in ["sentencepiece", "huggingface"]:
            raise ValueError(f"Unsupported format: {format}. Supported formats: ['sentencepiece', 'huggingface']")
        
        # Check if huggingface format is available
        if format == "huggingface" and not TRANSFORMERS_AVAILABLE:
            raise ImportError("HuggingFace format requires transformers and tokenizers libraries. "
                            "Install with: pip install transformers tokenizers")
        
        # Validate vocab_size
        supported_sizes = [8000, 16000, 32000, 64000]
        if vocab_size not in supported_sizes:
            raise ValueError(f"Unsupported vocab_size: {vocab_size}. "
                           f"Supported sizes: {supported_sizes}")
        
        # Load the tokenizer
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the tokenizer from HuggingFace."""
        model_filename = f"{self.lang}_tokenizer_{self.vocab_size//1000}k.model"
        
        # Load from HuggingFace
        try:
            model_file = hf_hub_download(
                repo_id=f"wikilangs/{self.lang}",
                filename=f"models/tokenizer/{model_filename}",
                repo_type="model"
            )
            # Initialize SentencePiece processor
            self.sp_tokenizer = spm.SentencePieceProcessor()
            self.sp_tokenizer.load(model_file)
            self._convert_to_format()
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load tokenizer for {self.lang} on {self.date} "
                f"with vocab size {self.vocab_size}: {str(e)}"
            )
    
    def _convert_to_format(self):
        """Convert the loaded SentencePiece tokenizer to the requested format."""
        if self.format == "sentencepiece":
            # Keep the original SentencePiece tokenizer
            self.tokenizer = self.sp_tokenizer
        elif self.format == "huggingface":
            # Convert to HuggingFace format
            self.tokenizer = self._convert_to_huggingface()
    
    def _convert_to_huggingface(self) -> PreTrainedTokenizerFast:
        """Convert SentencePiece tokenizer to HuggingFace format."""
        # Create vocabulary from SentencePiece
        vocab = {}
        for i in range(self.sp_tokenizer.get_piece_size()):
            piece = self.sp_tokenizer.id_to_piece(i)
            vocab[piece] = i
        
        # Create a BPE model for the tokenizer
        tokenizer_obj = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        
        # Set up pre-tokenizer (SentencePiece uses a specific pre-tokenizer)
        tokenizer_obj.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        
        # Set up decoder
        tokenizer_obj.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
        
        # Create special tokens mapping
        special_tokens = {
            "unk_token": self.sp_tokenizer.id_to_piece(self.sp_tokenizer.unk_id()),
            "bos_token": self.sp_tokenizer.id_to_piece(self.sp_tokenizer.bos_id()) if self.sp_tokenizer.bos_id() != -1 else None,
            "eos_token": self.sp_tokenizer.id_to_piece(self.sp_tokenizer.eos_id()) if self.sp_tokenizer.eos_id() != -1 else None,
            "pad_token": self.sp_tokenizer.id_to_piece(self.sp_tokenizer.pad_id()) if self.sp_tokenizer.pad_id() != -1 else None,
        }
        
        # Filter out None values
        special_tokens = {k: v for k, v in special_tokens.items() if v is not None}
        
        # Create HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            **special_tokens
        )
        
        return hf_tokenizer
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.
        
        Args:
            text (str): Text to encode
            
        Returns:
            List[int]: List of token IDs
        """
        if self.format == "sentencepiece":
            return self.tokenizer.encode(text, out_type=int)
        else:  # huggingface
            return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text.
        
        Args:
            token_ids (List[int]): List of token IDs
            
        Returns:
            str: Decoded text
        """
        return self.tokenizer.decode(token_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        if self.format == "sentencepiece":
            return self.tokenizer.encode(text, out_type=str)
        else:  # huggingface
            return self.tokenizer.tokenize(text)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value: int):
        """Set the vocabulary size."""
        self._vocab_size = value
    
    def get_vocab(self) -> dict:
        """Get the vocabulary mapping.
        
        Returns:
            dict: Vocabulary mapping (token -> ID)
        """
        if self.format == "sentencepiece":
            vocab = {}
            for i in range(self.tokenizer.get_piece_size()):
                piece = self.tokenizer.id_to_piece(i)
                vocab[piece] = i
            return vocab
        else:  # huggingface
            return self.tokenizer.get_vocab()


def tokenizer(lang: str, date: str = "latest", vocab_size: int = 16000, 
              format: Literal["sentencepiece", "huggingface"] = "sentencepiece") -> BPETokenizer:
    """Create a BPE tokenizer instance.
    
    Args:
        lang (str): Language code (e.g., 'en', 'fr', 'ary')
        date (str): Date of the model (format: YYYYMMDD, default: "latest")
        vocab_size (int): Vocabulary size (default: 16000)
        format (str): Output format - "sentencepiece" or "huggingface" (default: "sentencepiece")
        
    Returns:
        BPETokenizer: Initialized tokenizer instance
    """
    return BPETokenizer(lang=lang, date=date, vocab_size=vocab_size, format=format)
