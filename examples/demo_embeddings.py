#!/usr/bin/env python3
"""
Embedding operations demo using BabelVec format for the new repository structure.
"""

import sys
sys.path.insert(0, 'src')

from huggingface_hub import hf_hub_download
import json
import numpy as np

try:
    from babelvec import BabelVec
    BABELVEC_AVAILABLE = True
except ImportError:
    BABELVEC_AVAILABLE = False
    print("Warning: babelvec not installed. Install with: pip install babelvec")

def load_babelvec_embeddings(lang='ary', dimension=32):
    """Load BabelVec embeddings from the new repository structure."""
    print(f"Loading {dimension}D BabelVec embeddings for {lang}...")
    
    # Download embedding files
    embedding_file = hf_hub_download(
        repo_id=f'wikilangs/{lang}',
        filename=f'models/embeddings/monolingual/{lang}_{dimension}d.bin',
        repo_type='model'
    )
    
    metadata_file = hf_hub_download(
        repo_id=f'wikilangs/{lang}',
        filename=f'models/embeddings/monolingual/{lang}_{dimension}d_metadata.json',
        repo_type='model'
    )
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Embedding metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    if BABELVEC_AVAILABLE:
        # Load with BabelVec
        model = BabelVec.load(embedding_file)
        print(f"Model loaded successfully!")
        print(f"Is aligned: {model.is_aligned}")
        print(f"Vocabulary size: {len(model.words)}")
        return model, metadata
    else:
        print("BabelVec not available, returning file paths")
        return embedding_file, metadata

def demo_embedding_operations():
    """Demonstrate embedding operations."""
    print("=" * 60)
    print("BABELVEC EMBEDDING OPERATIONS DEMO")
    print("=" * 60)
    
    if not BABELVEC_AVAILABLE:
        print("BabelVec not installed. Install with: pip install babelvec")
        return
    
    # Load embeddings
    model, metadata = load_babelvec_embeddings('ary', 32)
    
    print("\n1. WORD VECTOR OPERATIONS")
    print("-" * 30)
    
    # Test words
    test_words = ['hello', 'world', 'مرحبا', 'اختبار', 'language']
    
    for word in test_words:
        try:
            vector = model.embed_word(word)
            if vector is not None:
                print(f"'{word}': vector shape {vector.shape}, norm: {np.linalg.norm(vector):.4f}")
            else:
                print(f"'{word}': not in vocabulary")
        except Exception as e:
            print(f"'{word}': error - {e}")
    
    print("\n2. SENTENCE EMBEDDINGS")
    print("-" * 30)
    
    test_sentences = [
        "hello world",
        "this is a test",
        "مرحبا بالعالم",
        "the quick brown fox jumps over the lazy dog"
    ]
    
    methods = ['average', 'rope', 'decay', 'sinusoidal']
    
    for sentence in test_sentences:
        print(f"\nSentence: '{sentence}'")
        for method in methods:
            try:
                vector = model.embed_sentence(sentence, method=method)
                print(f"  {method:12}: shape {vector.shape}, norm: {np.linalg.norm(vector):.4f}")
            except Exception as e:
                print(f"  {method:12}: error - {e}")
    
    print("\n3. SIMILARITY OPERATIONS")
    print("-" * 30)
    
    # Test word similarities
    word_pairs = [
        ('hello', 'world'),
        ('test', 'exam'),
        ('مرحبا', 'أهلا'),
    ]
    
    for word1, word2 in word_pairs:
        try:
            vec1 = model.embed_word(word1)
            vec2 = model.embed_word(word2)
            
            if vec1 is not None and vec2 is not None:
                # Cosine similarity
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                print(f"'{word1}' vs '{word2}': cosine similarity = {cos_sim:.4f}")
            else:
                print(f"'{word1}' vs '{word2}': one or both words not in vocabulary")
        except Exception as e:
            print(f"'{word1}' vs '{word2}': error - {e}")
    
    print("\n4. POSITION-AWARE ENCODING COMPARISON")
    print("-" * 40)
    
    # Test position-aware vs average encoding
    sentences = [
        "the dog bites the man",
        "the man bites the dog",
        "dog bites man the"
    ]
    
    for sentence in sentences:
        try:
            avg_vec = model.embed_sentence(sentence, method='average')
            rope_vec = model.embed_sentence(sentence, method='rope')
            
            # Compare average vs position-aware
            similarity = np.dot(avg_vec, rope_vec) / (np.linalg.norm(avg_vec) * np.linalg.norm(rope_vec))
            print(f"'{sentence}'")
            print(f"  Average vs RoPE similarity: {similarity:.4f}")
        except Exception as e:
            print(f"'{sentence}': error - {e}")
    
    print("\n5. VOCABULARY ANALYSIS")
    print("-" * 25)
    
    try:
        vocab_words = list(model.words)
        print(f"Total vocabulary size: {len(vocab_words):,}")
        
        # Sample some vocabulary words
        sample_words = vocab_words[:20]
        print(f"Sample words: {sample_words}")
        
        # Vector statistics
        sample_vectors = []
        for word in sample_words[:10]:
            vec = model.embed_word(word)
            if vec is not None:
                sample_vectors.append(vec)
        
        if sample_vectors:
            all_vectors = np.vstack(sample_vectors)
            print(f"Vector statistics (sample of {len(sample_vectors)} words):")
            print(f"  Mean norm: {np.mean(np.linalg.norm(all_vectors, axis=1)):.4f}")
            print(f"  Std norm:  {np.std(np.linalg.norm(all_vectors, axis=1)):.4f}")
            print(f"  Mean values: {np.mean(all_vectors):.4f}")
            print(f"  Std values:  {np.std(all_vectors):.4f}")
        
    except Exception as e:
        print(f"Vocabulary analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("EMBEDDING DEMO COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    demo_embedding_operations()
