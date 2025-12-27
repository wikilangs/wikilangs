"""Basic usage examples for the wikilangs package."""

from wikilangs import tokenizer, ngram, markov, vocabulary


def example_tokenizer():
    """Example of using the tokenizer."""
    print("=== Tokenizer Example ===")
    
    # Create a tokenizer for Moroccan Arabic (date defaults to 'latest')
    tok = tokenizer(lang='ary', vocab_size=16000)
    
    # Tokenize some Arabic text
    text = "مرحبا بك فالمغرب العربي"
    tokens = tok.tokenize(text)
    token_ids = tok.encode(text)
    
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {tok.decode(token_ids)}")
    print()

def example_ngram():
    """Example of using the n-gram model."""
    print("=== N-gram Model Example ===")
    
    # Create a 3-gram model for Moroccan Arabic
    ng = ngram(lang='ary', gram_size=3)
    
    # Score Arabic text using tokenized format
    from wikilangs import tokenizer
    tok = tokenizer(lang='ary', vocab_size=16000)
    
    # Use Arabic text and tokenize it first
    text = "فالمغرب العربي"
    tokens = tok.tokenize(text)
    tokenized_text = " ".join(tokens)
    score = ng.score(tokenized_text)
    
    print(f"Original text: {text}")
    print(f"Tokenized text: {tokenized_text}")
    print(f"Score: {score}")
    
    # Predict next tokens using tokenized context
    if len(tokens) >= 2:
        context_tokens = tokens[:2]
        context = " ".join(context_tokens)
        predictions = ng.predict_next(context, top_k=3)
        
        print(f"Context: {context}")
        print(f"Predictions: {predictions}")
    else:
        print("Not enough tokens for prediction")
    print()

def example_markov():
    """Example of using the Markov chain."""
    print("=== Markov Chain Example ===")
    
    # Create a Markov chain for Moroccan Arabic
    mc = markov(lang='ary', depth=2)
    
    # Generate text
    generated_text = mc.generate(length=20)
    
    print(f"Generated text: {generated_text}")
    
    # Generate with Arabic seed
    from wikilangs import tokenizer
    tok = tokenizer(lang='ary', vocab_size=16000)
    seed_text = "فالمغرب"
    seed_tokens = tok.tokenize(seed_text)
    generated_with_seed = mc.generate(length=15, seed=seed_tokens)
    
    print(f"Seed: {seed_text}")
    print(f"Generated with seed: {generated_with_seed}")
    print()

def example_vocabulary():
    """Example of using the vocabulary."""
    print("=== Vocabulary Example ===")
    
    # Create a vocabulary for Moroccan Arabic
    vocab = vocabulary(lang='ary')
    
    # Look up Arabic words
    sample_words = list(vocab.words)[:5]  # Get some sample words
    
    if sample_words:
        word = sample_words[0]
        word_info = vocab.lookup(word)
        frequency = vocab.get_frequency(word)
        similar_words = vocab.get_similar_words(word, top_k=3)
        
        # Try prefix search with first 2 characters of the word
        if len(word) >= 2:
            prefix = word[:2]
            prefix_words = vocab.get_words_with_prefix(prefix, top_k=5)
        else:
            prefix = word
            prefix_words = vocab.get_words_with_prefix(prefix, top_k=5)
        
        print(f"Sample word: {word}")
        print(f"Information: {word_info}")
        print(f"Frequency: {frequency}")
        print(f"Similar words: {similar_words}")
        print(f"Words with prefix '{prefix}': {prefix_words[:3]}")
        print(f"Vocabulary size: {vocab.size}")
    else:
        print("No words found in vocabulary")
    print()

def main():
    """Run all examples."""
    print("Wikilangs Package Examples\n")
    
    try:
        example_tokenizer()
    except Exception as e:
        print(f"Tokenizer example failed: {e}\n")
    
    try:
        example_ngram()
    except Exception as e:
        print(f"N-gram example failed: {e}\n")
    
    try:
        example_markov()
    except Exception as e:
        print(f"Markov example failed: {e}\n")
    
    try:
        example_vocabulary()
    except Exception as e:
        print(f"Vocabulary example failed: {e}\n")


if __name__ == "__main__":
    main()
