"""Tests for the embeddings module.

These tests use real data from HuggingFace and will fail if the models don't exist
or if there are actual issues with loading/using the embeddings.
"""

import pytest
import numpy as np
from wikilangs.embeddings import Embeddings, embeddings, BABELVEC_AVAILABLE


class TestEmbeddings:
    """Test cases for the Embeddings class using real data."""
    
    @pytest.mark.skipif(not BABELVEC_AVAILABLE, reason="babelvec not installed")
    def test_load_real_embeddings_ary(self):
        """Test loading real embeddings for Moroccan Arabic."""
        # This will download from HF if not local
        emb = embeddings(lang='ary', dimension=32)
        
        # Check that it's an Embeddings instance (not the fallback tuple)
        assert isinstance(emb, Embeddings)
        assert emb.metadata['language'] == 'ary'
        assert emb.dimension == 32
        
        # Test word vector
        word = "مرحبا"
        vec = emb.embed_word(word)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (32,)
        
        # Test sentence vector
        sentence = "مرحبا بالعالم"
        sent_vec = emb.embed_sentence(sentence, method='rope')
        assert isinstance(sent_vec, np.ndarray)
        assert sent_vec.shape == (32,)
        
        # Test words property
        words = emb.words
        assert isinstance(words, list)
        assert len(words) > 0
        assert word in words or any(word in w for w in words[:100]) # Basic check

    def test_embeddings_fallback_when_no_babelvec(self, monkeypatch):
        """Test that the factory function returns a tuple when babelvec is missing."""
        import sys
        from wikilangs.embeddings import Embeddings
        module = sys.modules[Embeddings.__module__]
        monkeypatch.setattr(module, "BABELVEC_AVAILABLE", False)
        
        # This should return (file_path, metadata)
        from wikilangs.embeddings import embeddings as patched_embeddings
        result = patched_embeddings(date='20251201', lang='ary', dimension=32)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str) # file path
        assert isinstance(result[1], dict) # metadata
        assert result[1]['language'] == 'ary'

    def test_load_nonexistent_embeddings(self):
        """Test loading embeddings for a nonexistent language."""
        with pytest.raises(FileNotFoundError):
            embeddings(date='20251201', lang='nonexistent_lang', dimension=32)

    @pytest.mark.skipif(not BABELVEC_AVAILABLE, reason="babelvec not installed")
    def test_embeddings_methods(self):
        """Test different sentence embedding methods."""
        emb = embeddings(date='20251201', lang='ary', dimension=32)
        assert isinstance(emb, Embeddings)
        sentence = "test sentence"
        
        methods = ['average', 'rope', 'decay', 'sinusoidal']
        for method in methods:
            vec = emb.embed_sentence(sentence, method=method)
            assert vec.shape == (32,)
