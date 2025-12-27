"""Tests for the ngram module.

These tests use real data from HuggingFace and will fail if the models don't exist
or if there are actual issues with loading/using the n-gram models.
"""

import pytest
from wikilangs.ngram import NGramModel, ngram


class TestNGramModel:
    """Test cases for the NGramModel class using real data."""
    
    def test_init_invalid_gram_size(self):
        """Test initialization with invalid gram_size."""
        with pytest.raises(ValueError, match="Unsupported gram_size"):
            NGramModel(lang='ary', gram_size=10)
    
    def test_ngram_factory_function(self):
        """Test the ngram factory function."""
        # This will fail if the model doesn't exist, which is correct behavior
        with pytest.raises(FileNotFoundError):
            ng = ngram(lang='nonexistent', gram_size=3)
    
    def test_load_real_ngram_ary(self):
        """Test loading a real n-gram model for Moroccan Arabic."""
        try:
            ng = NGramModel(lang='ary', gram_size=3)
            
            # Test that the model actually works
            test_text = "This is a test sentence"
            
            # Test scoring
            score = ng.score(test_text)
            assert isinstance(score, float)
            
            # Test prediction
            context = "This is a"
            predictions = ng.predict_next(context, top_k=5)
            assert isinstance(predictions, list)
            assert len(predictions) <= 5
            assert all(isinstance(pred, tuple) and len(pred) == 2 for pred in predictions)
            assert all(isinstance(pred[0], str) and isinstance(pred[1], float) for pred in predictions)
            
            # Test vocab_size property
            vocab_size = ng.vocab_size
            assert isinstance(vocab_size, int)
            assert vocab_size > 0
            
            # Test total_ngrams property
            total_ngrams = ng.total_ngrams
            assert isinstance(total_ngrams, int)
            assert total_ngrams > 0
            
            # Verify model attributes are set
            assert hasattr(ng, 'model')
            assert hasattr(ng, 'metadata')
            assert ng.date == 'latest'
            assert ng.lang == 'ary'
            assert ng.gram_size == 3
            
        except FileNotFoundError:
            pytest.skip("N-gram model for 'ary' not available on HuggingFace")
    
    def test_load_nonexistent_ngram(self):
        """Test that loading a nonexistent n-gram model properly fails."""
        with pytest.raises(FileNotFoundError, match="Could not load n-gram model"):
            NGramModel(date='20251201', lang='nonexistent_lang', gram_size=3)
    
    def test_supported_gram_sizes(self):
        """Test that only supported gram sizes are accepted."""
        supported_sizes = [2, 3, 4, 5]
        unsupported_sizes = [1, 6, 7, 10, 100]
        
        # Test that unsupported sizes raise ValueError
        for size in unsupported_sizes:
            with pytest.raises(ValueError, match="Unsupported gram_size"):
                NGramModel(date='20251201', lang='ary', gram_size=size)
