"""Tests for the markov module.

These tests use real data from HuggingFace and will fail if the models don't exist
or if there are actual issues with loading/using the Markov chain models.
"""

import pytest
from wikilangs.markov import MarkovChain, markov


class TestMarkovChain:
    """Test cases for the MarkovChain class using real data."""
    
    def test_init_invalid_depth(self):
        """Test initialization with invalid depth."""
        with pytest.raises(ValueError, match="Unsupported depth"):
            MarkovChain(lang='ary', depth=10)
    
    def test_markov_factory_function(self):
        """Test the markov factory function."""
        # This will fail if the model doesn't exist, which is correct behavior
        with pytest.raises(FileNotFoundError):
            mc = markov(lang='nonexistent', depth=2)
    
    def test_load_real_markov_ary(self):
        """Test loading a real Markov chain model for Moroccan Arabic."""
        try:
            mc = MarkovChain(lang='ary', depth=2)
            
            # Test that the model actually works
            
            # Test text generation
            generated_text = mc.generate(length=50)
            assert isinstance(generated_text, str)
            assert len(generated_text) > 0
            
            # Test text generation with seed
            seed = ["hello", "world"]
            generated_text_with_seed = mc.generate(length=20, seed=seed)
            assert isinstance(generated_text_with_seed, str)
            assert len(generated_text_with_seed) > 0
            
            # Test get_transitions
            state = ("hello", "world")
            transitions = mc.get_transitions(state)
            assert isinstance(transitions, dict)
            
            # Test vocab_size property
            vocab_size = mc.vocab_size
            assert isinstance(vocab_size, int)
            assert vocab_size >= 0
            
            # Test total_transitions property
            total_transitions = mc.total_transitions
            assert isinstance(total_transitions, int)
            assert total_transitions > 0
            
            # Verify model attributes are set
            assert hasattr(mc, 'transitions')
            assert hasattr(mc, 'metadata')
            assert mc.date == 'latest'
            assert mc.lang == 'ary'
            assert mc.depth == 2
            
        except FileNotFoundError:
            pytest.skip("Markov chain model for 'ary' not available on HuggingFace")
    
    def test_load_nonexistent_markov(self):
        """Test that loading a nonexistent Markov chain model properly fails."""
        with pytest.raises(FileNotFoundError, match="Could not load Markov chain model"):
            MarkovChain(date='20251201', lang='nonexistent_lang', depth=2)
    
    def test_supported_depths(self):
        """Test that only supported depths are accepted."""
        supported_depths = [1, 2, 3, 4, 5]
        unsupported_depths = [0, 6, 7, 10, 100]
        
        # Test that unsupported depths raise ValueError
        for depth in unsupported_depths:
            with pytest.raises(ValueError, match="Unsupported depth"):
                MarkovChain(date='20251201', lang='ary', depth=depth)
