"""Basic import tests for the wikilangs package."""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import wikilangs


def test_module_imports():
    """Test that all modules can be imported."""
    from wikilangs import tokenizer, ngram, markov, vocabulary
    
    # Check that the modules exist
    assert tokenizer is not None
    assert ngram is not None
    assert markov is not None
    assert vocabulary is not None


def test_factory_functions():
    """Test that factory functions exist."""
    from wikilangs.tokenizer import tokenizer as tokenizer_factory
    from wikilangs.ngram import ngram as ngram_factory
    from wikilangs.markov import markov as markov_factory
    from wikilangs.vocabulary import vocabulary as vocabulary_factory
    
    # Check that the factory functions exist
    assert tokenizer_factory is not None
    assert ngram_factory is not None
    assert markov_factory is not None
    assert vocabulary_factory is not None
