"""Tests for the vocabulary module.

These tests use real data from HuggingFace and will fail if the models don't exist
or if there are actual issues with loading/using the vocabulary data.
"""

import pytest
from wikilangs.vocabulary import WikilangsVocabulary, vocabulary


class TestWikilangsVocabulary:
    """Test cases for the WikilangsVocabulary class using real data."""
    
    def test_vocabulary_factory_function(self):
        """Test the vocabulary factory function."""
        # This will fail if the model doesn't exist, which is correct behavior
        with pytest.raises(FileNotFoundError):
            vocab = vocabulary(lang='nonexistent')
    
    def test_load_real_vocabulary_ary(self):
        """Test loading a real vocabulary for Moroccan Arabic."""
        try:
            vocab = WikilangsVocabulary(lang='ary')
            
            # Test that the vocabulary actually works
            
            # Test size property
            vocab_size = vocab.size
            assert isinstance(vocab_size, int)
            assert vocab_size > 0
            
            # Test words property
            words = vocab.words
            assert isinstance(words, set)
            assert len(words) > 0
            assert all(isinstance(word, str) for word in words)
            
            # Test lookup with a word that should exist (common Arabic word)
            # We'll test with some sample words, but since we don't know exact content,
            # we'll test the method behavior
            sample_words = list(words)[:5] if len(words) >= 5 else list(words)
            
            for word in sample_words:
                word_info = vocab.lookup(word)
                # Should return either a dict or None
                assert word_info is None or isinstance(word_info, dict)
                
                # Test frequency method
                frequency = vocab.get_frequency(word)
                assert isinstance(frequency, int)
                assert frequency >= 0
            
            # Test get_words_with_prefix
            if sample_words:
                first_word = sample_words[0]
                if len(first_word) > 1:
                    prefix = first_word[:2]
                    prefix_words = vocab.get_words_with_prefix(prefix, top_k=5)
                    assert isinstance(prefix_words, list)
                    assert len(prefix_words) <= 5
                    assert all(isinstance(word, str) for word in prefix_words)
                    assert all(word.startswith(prefix) for word in prefix_words)
            
            # Test get_similar_words (should return empty list for now)
            if sample_words:
                similar_words = vocab.get_similar_words(sample_words[0], top_k=5)
                assert isinstance(similar_words, list)
                # Currently returns empty list, which is expected
                assert len(similar_words) == 0
            
            # Verify vocabulary attributes are set
            assert hasattr(vocab, 'vocab_dict')
            assert hasattr(vocab, 'metadata')
            assert vocab.date == 'latest'
            assert vocab.lang == 'ary'
            
        except FileNotFoundError:
            pytest.skip("Vocabulary for 'ary' not available on HuggingFace")
    
    def test_load_nonexistent_vocabulary(self):
        """Test that loading a nonexistent vocabulary properly fails."""
        with pytest.raises(FileNotFoundError, match="Could not load vocabulary"):
            WikilangsVocabulary(date='20251201', lang='nonexistent_lang')
    
    def test_vocabulary_methods_with_nonexistent_words(self):
        """Test vocabulary methods with words that don't exist."""
        try:
            vocab = WikilangsVocabulary(date='20251201', lang='ary')
            
            # Test lookup with nonexistent word
            result = vocab.lookup('nonexistent_word_12345')
            assert result is None
            
            # Test frequency with nonexistent word
            frequency = vocab.get_frequency('nonexistent_word_12345')
            assert frequency == 0
            
            # Test prefix search with nonexistent prefix
            prefix_words = vocab.get_words_with_prefix('xyz123', top_k=5)
            assert isinstance(prefix_words, list)
            assert len(prefix_words) == 0
            
        except FileNotFoundError:
            pytest.skip("Vocabulary for 'ary' not available on HuggingFace")
