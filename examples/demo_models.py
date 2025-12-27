#!/usr/bin/env python3
"""
Demo script showcasing Wikilangs model operations with the new HuggingFace repository structure.
"""

import sys
sys.path.insert(0, 'src')

from wikilangs import vocabulary, tokenizer, ngram, markov, languages

def main():
    print("=" * 60)
    print("WIKILANGS MODEL DEMO - New Repository Structure")
    print("=" * 60)
    
    # Test 1: Available languages
    print("\n1. AVAILABLE LANGUAGES")
    print("-" * 30)
    available_langs = languages()
    print(f"Languages available: {available_langs}")
    
    # Test 2: Tokenizer operations
    print("\n2. TOKENIZER OPERATIONS")
    print("-" * 30)
    print("Loading tokenizer...")
    tok = tokenizer(lang='ary', vocab_size=8000)
    
    test_texts = [
        "hello world",
        "مرحبا بالعالم",  # "Hello world" in Arabic
        "this is a test sentence"
    ]
    
    for text in test_texts:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        print(f"Text: '{text}'")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Token count: {len(encoded)}")
        print()
    
    # Test 3: Vocabulary operations
    print("3. VOCABULARY OPERATIONS")
    print("-" * 30)
    print("Loading vocabulary...")
    vocab = vocabulary(lang='ary')
    
    print(f"Vocabulary size: {vocab.size:,} words")
    
    # Test word lookup
    test_words = ["hello", "world", "مرحبا", "اختبار"]
    print("\nWord lookups:")
    for word in test_words:
        word_info = vocab.lookup(word)
        frequency = vocab.get_frequency(word)
        print(f"  '{word}': freq={frequency:,}, info={word_info is not None}")
    
    # Test prefix search
    print("\nPrefix search for 'he':")
    prefix_words = vocab.get_words_with_prefix('he', top_k=5)
    for word in prefix_words:
        print(f"  {word} (freq: {vocab.get_frequency(word):,})")
    
    # Test 4: N-gram operations
    print("\n4. N-GRAM OPERATIONS")
    print("-" * 30)
    print("Loading 2-gram model...")
    ng = ngram(lang='ary', gram_size=2)
    
    print(f"N-gram vocabulary size: {ng.vocab_size:,}")
    print(f"Total n-grams: {ng.total_ngrams:,}")
    
    # Test scoring
    test_sentences = [
        "hello world",
        "this is a test",
        "مرحبا بالعالم"
    ]
    
    print("\nSentence scores (log probabilities):")
    for sentence in test_sentences:
        score = ng.score(sentence)
        print(f"  '{sentence}': {score:.4f}")
    
    # Test prediction
    print("\nNext word predictions:")
    contexts = ["hello", "this is", "مرحبا"]
    for context in contexts:
        predictions = ng.predict_next(context, top_k=3)
        print(f"  Context: '{context}'")
        for word, prob in predictions:
            print(f"    {word}: {prob:.4f}")
    
    # Test 5: Markov chain operations
    print("\n5. MARKOV CHAIN OPERATIONS")
    print("-" * 30)
    print("Loading Markov chain (depth=2)...")
    mk = markov(lang='ary', depth=2)
    
    print(f"Markov chain depth: {mk.depth}")
    print(f"Unique contexts: {mk.metadata.get('unique_contexts', 'N/A')}")
    print(f"Total transitions: {mk.metadata.get('total_transitions', 'N/A')}")
    
    # Test text generation
    print("\nText generation:")
    seeds = ["hello", "this", "مرحبا"]
    for seed in seeds:
        generated = mk.generate(length=10, seed=[seed] if seed else None)
        print(f"  Seed '{seed}': '{generated}'")
    
    # Test 6: Combined operations
    print("\n6. COMBINED OPERATIONS")
    print("-" * 30)
    print("Demonstrating pipeline: Tokenize → Generate → Score")
    
    # Start with a seed
    seed_text = "hello"
    print(f"Starting with: '{seed_text}'")
    
    # Tokenize
    tokens = tok.encode(seed_text)
    print(f"Tokenized: {tokens}")
    
    # Generate continuation using Markov
    continuation = mk.generate(length=5, seed=[seed_text])
    full_text = seed_text + " " + continuation
    print(f"Generated: '{full_text}'")
    
    # Score the result
    score = ng.score(full_text)
    print(f"N-gram score: {score:.4f}")
    
    # Get vocabulary stats for generated words
    generated_words = full_text.split()
    vocab_coverage = sum(1 for word in generated_words if vocab.lookup(word)) / len(generated_words)
    print(f"Vocabulary coverage: {vocab_coverage:.2%}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("All models are working with the new repository structure.")
    print("=" * 60)

if __name__ == "__main__":
    main()
