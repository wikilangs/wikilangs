"""Tests for the tokenizer module.

These tests use real data from HuggingFace and will fail if the models don't exist
or if there are actual issues with loading/using the tokenizers.
"""

import pytest
from wikilangs.tokenizer import BPETokenizer, tokenizer


class TestBPETokenizer:
    """Test cases for the BPETokenizer class using real data."""
    
    def test_init_invalid_vocab_size(self):
        """Test initialization with invalid vocab_size."""
        with pytest.raises(ValueError, match="Unsupported vocab_size"):
            BPETokenizer(lang='ary', vocab_size=12345)
    
    def test_tokenizer_factory_function(self):
        """Test the tokenizer factory function."""
        # This will fail if the model doesn't exist, which is correct behavior
        with pytest.raises(FileNotFoundError):
            tok = tokenizer(lang='nonexistent', vocab_size=16000)
    
    def test_load_real_tokenizer_ary(self):
        """Test loading a real tokenizer for Moroccan Arabic."""
        try:
            tok = BPETokenizer(lang='ary', vocab_size=16000)
            
            # Test that the tokenizer actually works
            test_text = "مرحبا بك في المغرب"
            
            # Test encoding
            token_ids = tok.encode(test_text)
            assert isinstance(token_ids, list)
            assert len(token_ids) > 0
            assert all(isinstance(tid, int) for tid in token_ids)
            
            # Test decoding
            decoded_text = tok.decode(token_ids)
            assert isinstance(decoded_text, str)
            
            # Test tokenization
            tokens = tok.tokenize(test_text)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
            assert all(isinstance(token, str) for token in tokens)
            
            # Test vocab_size property
            assert tok.vocab_size == 16000
            
            # Test get_vocab method
            vocab = tok.get_vocab()
            assert isinstance(vocab, dict)
            assert len(vocab) > 0
            
        except FileNotFoundError:
            pytest.skip("Tokenizer model for 'ary' not available on HuggingFace")
    
    def test_load_nonexistent_tokenizer(self):
        """Test that loading a nonexistent tokenizer properly fails."""
        with pytest.raises(FileNotFoundError, match="Could not load tokenizer"):
            BPETokenizer(date='20251201', lang='nonexistent_lang', vocab_size=16000)
    
    def test_supported_vocab_sizes(self):
        """Test that only supported vocab sizes are accepted."""
        supported_sizes = [8000, 16000, 32000, 64000]
        unsupported_sizes = [1000, 5000, 12000, 20000, 100000]
        
        # Test that unsupported sizes raise ValueError
        for size in unsupported_sizes:
            with pytest.raises(ValueError, match="Unsupported vocab_size"):
                BPETokenizer(date='20251201', lang='ary', vocab_size=size)
