#!/usr/bin/env python3
"""
Comprehensive integration demo showing all Wikilangs models working together
with the new HuggingFace repository structure and BabelVec embeddings.
"""

import sys
sys.path.insert(0, 'src')

from wikilangs import vocabulary, tokenizer, ngram, markov, languages
from huggingface_hub import hf_hub_download
import json
import numpy as np

try:
    from babelvec import BabelVec
    BABELVEC_AVAILABLE = True
except ImportError:
    BABELVEC_AVAILABLE = False
    print("Warning: babelvec not installed. Install with: pip install babelvec")

def load_all_models(lang='ary'):
    """Load all available models for a language."""
    print(f"Loading all models for {lang}...")
    
    models = {}
    
    # Load core models
    models['vocabulary'] = vocabulary(date='20251201', lang=lang)
    models['tokenizer'] = tokenizer(date='20251201', lang=lang, vocab_size=8000)
    models['ngram_2'] = ngram(date='20251201', lang=lang, gram_size=2)
    models['ngram_3'] = ngram(date='20251201', lang=lang, gram_size=3)
    models['markov_2'] = markov(date='20251201', lang=lang, depth=2)
    
    # Load embeddings if available
    if BABELVEC_AVAILABLE:
        try:
            embedding_file = hf_hub_download(
                repo_id=f'wikilangs/{lang}',
                filename=f'models/embeddings/monolingual/{lang}_32d.bin',
                repo_type='model'
            )
            models['embeddings'] = BabelVec.load(embedding_file)
            print("Embeddings loaded successfully!")
        except Exception as e:
            print(f"Could not load embeddings: {e}")
            models['embeddings'] = None
    else:
        models['embeddings'] = None
    
    return models

def analyze_text_comprehensive(models, text):
    """Comprehensive text analysis using all available models."""
    print(f"\nAnalyzing: '{text}'")
    print("-" * 50)
    
    # 1. Tokenization
    tokens = models['tokenizer'].encode(text)
    decoded = models['tokenizer'].decode(tokens)
    print(f"Tokenization: {len(tokens)} tokens")
    print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    
    # 2. Vocabulary coverage
    words = text.split()
    vocab_coverage = sum(1 for word in words if models['vocabulary'].lookup(word)) / len(words)
    print(f"Vocabulary coverage: {vocab_coverage:.1%} ({sum(1 for word in words if models['vocabulary'].lookup(word))}/{len(words)} words)")
    
    # 3. N-gram scoring
    score_2 = models['ngram_2'].score(text)
    score_3 = models['ngram_3'].score(text)
    print(f"2-gram score: {score_2:.2f}")
    print(f"3-gram score: {score_3:.2f}")
    
    # 4. Word predictions
    if len(words) >= 2:
        context = ' '.join(words[-2:])
        predictions = models['ngram_2'].predict_next(context, top_k=3)
        print(f"Next word predictions for '{context}':")
        for word, prob in predictions:
            print(f"  {word}: {prob:.4f}")
    
    # 5. Embeddings analysis
    if models['embeddings']:
        try:
            # Get sentence embedding
            sent_vec_avg = models['embeddings'].embed_sentence(text, method='average')
            sent_vec_rope = models['embeddings'].embed_sentence(text, method='rope')
            
            # Word embeddings for individual words
            word_embeddings = {}
            for word in words[:5]:  # First 5 words
                vec = models['embeddings'].embed_word(word)
                if vec is not None:
                    word_embeddings[word] = vec
            
            print(f"Embeddings analysis:")
            print(f"  Sentence vector (avg): norm {np.linalg.norm(sent_vec_avg):.4f}")
            print(f"  Sentence vector (RoPE): norm {np.linalg.norm(sent_vec_rope):.4f}")
            print(f"  Word embeddings found: {len(word_embeddings)}")
            
            # Calculate similarities between words
            if len(word_embeddings) >= 2:
                word_pairs = list(word_embeddings.items())[:3]
                for i, (word1, vec1) in enumerate(word_pairs):
                    for word2, vec2 in word_pairs[i+1:i+2]:
                        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        print(f"    '{word1}' vs '{word2}': {cos_sim:.4f}")
        except Exception as e:
            print(f"Embeddings error: {e}")

def generate_and_analyze(models, seed, length=10):
    """Generate text using Markov and analyze with all models."""
    print(f"\nGenerating from seed: '{seed}'")
    print("-" * 40)
    
    # Generate text
    generated = models['markov_2'].generate(length=length, seed=[seed] if seed else None)
    print(f"Generated: '{generated}'")
    
    # Analyze the generated text
    analyze_text_comprehensive(models, generated)

def compare_languages():
    """Compare available languages if multiple are available."""
    print("\nLANGUAGE COMPARISON")
    print("=" * 50)
    
    available_langs = languages(date='20251201')
    print(f"Available languages: {available_langs}")
    
    if len(available_langs) > 1:
        # Compare vocabulary sizes
        print("\nVocabulary size comparison:")
        for lang in available_langs:
            try:
                vocab = vocabulary(date='20251201', lang=lang)
                print(f"  {lang}: {vocab.size:,} words")
            except Exception as e:
                print(f"  {lang}: error - {e}")
    else:
        print("Only one language available for comparison")

def main():
    print("=" * 60)
    print("COMPREHENSIVE WIKILANGS INTEGRATION DEMO")
    print("New Repository Structure + BabelVec Embeddings")
    print("=" * 60)
    
    # 1. Language comparison
    compare_languages()
    
    # 2. Load all models
    models = load_all_models('ary')
    
    # 3. Model statistics
    print(f"\nMODEL STATISTICS")
    print("-" * 30)
    print(f"Vocabulary size: {models['vocabulary'].size:,}")
    print(f"Tokenizer vocab size: {models['tokenizer'].vocab_size}")
    print(f"2-gram model: {models['ngram_2'].vocab_size:,} vocab, {models['ngram_2'].total_ngrams:,} n-grams")
    print(f"3-gram model: {models['ngram_3'].vocab_size:,} vocab, {models['ngram_3'].total_ngrams:,} n-grams")
    print(f"Markov chain: depth {models['markov_2'].depth}")
    if models['embeddings']:
        print(f"Embeddings: {len(models['embeddings'].words):,} words, aligned: {models['embeddings'].is_aligned}")
    
    # 4. Text analysis examples
    print(f"\nTEXT ANALYSIS EXAMPLES")
    print("=" * 50)
    
    test_texts = [
        "hello world",
        "this is a test sentence",
        "مرحبا بالعالم",
        "the quick brown fox jumps over the lazy dog"
    ]
    
    for text in test_texts:
        analyze_text_comprehensive(models, text)
    
    # 5. Text generation and analysis
    print(f"\nTEXT GENERATION AND ANALYSIS")
    print("=" * 50)
    
    seeds = ["hello", "this", "مرحبا", "the"]
    for seed in seeds:
        generate_and_analyze(models, seed, length=8)
    
    # 6. Cross-model integration example
    print(f"\nCROSS-MODEL INTEGRATION EXAMPLE")
    print("=" * 50)
    
    # Start with a seed
    seed = "language"
    print(f"Starting with: '{seed}'")
    
    # Generate with Markov
    generated = models['markov_2'].generate(length=6, seed=[seed])
    print(f"Markov generated: '{generated}'")
    
    # Analyze with all models
    tokens = models['tokenizer'].encode(generated)
    words = generated.split()
    vocab_words = [w for w in words if models['vocabulary'].lookup(w)]
    
    score_2 = models['ngram_2'].score(generated)
    score_3 = models['ngram_3'].score(generated)
    
    print(f"Analysis:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Words: {len(words)}")
    print(f"  Vocabulary coverage: {len(vocab_words)}/{len(words)} ({len(vocab_words)/len(words):.1%})")
    print(f"  2-gram score: {score_2:.2f}")
    print(f"  3-gram score: {score_3:.2f}")
    
    # Embeddings if available
    if models['embeddings']:
        try:
            sent_vec = models['embeddings'].embed_sentence(generated, method='rope')
            print(f"  Embedding norm: {np.linalg.norm(sent_vec):.4f}")
            
            # Find most similar words in generated text
            word_sims = []
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    vec1 = models['embeddings'].embed_word(word1)
                    vec2 = models['embeddings'].embed_word(word2)
                    if vec1 is not None and vec2 is not None:
                        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        word_sims.append((word1, word2, sim))
            
            if word_sims:
                word_sims.sort(key=lambda x: x[2], reverse=True)
                print(f"  Most similar word pair: '{word_sims[0][0]}' & '{word_sims[0][1]}' ({word_sims[0][2]:.4f})")
        except Exception as e:
            print(f"  Embeddings analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE DEMO COMPLETED!")
    print("All models successfully integrated with new repository structure!")
    print("=" * 60)

if __name__ == "__main__":
    main()
