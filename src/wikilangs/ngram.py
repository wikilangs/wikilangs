"""N-gram model module for Wikilangs.

This module provides functionality to load and use n-gram language models
trained on Wikipedia data for different languages and dates.
"""

import os
import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError


class NGramModel:
    """N-gram language model for scoring and predicting text."""
    
    def __init__(self, lang: str, date: str = "latest", gram_size: int = 3, variant: str = "word"):
        """Initialize the n-gram model.
        
        Args:
            lang (str): Language code (e.g., 'en', 'fr', 'ary')
            date (str): Date of the model (format: YYYYMMDD, default: "latest")
            gram_size (int): Size of n-grams (default: 3)
            variant (str, optional): Model variant (default: "word", can be "subword")
        
        Raises:
            FileNotFoundError: If the model files cannot be found
            ValueError: If gram_size is not supported
        """
        self.date = date
        self.lang = lang
        self.gram_size = gram_size
        self.resolved_date = date
        self.variant = variant  # "word" or "subword"
        self.repo_id = f"wikilangs/{self.lang}"
        
        self.model = None
        self.metadata = None
        
        # Validate gram_size
        supported_sizes = [2, 3, 4, 5]
        if gram_size not in supported_sizes:
            raise ValueError(f"Unsupported gram_size: {gram_size}. "
                           f"Supported sizes: {supported_sizes}")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the n-gram model from HuggingFace."""
        model_file, metadata_file = self._download_model_from_hf()
        
        self.model = pd.read_parquet(model_file)
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

    def _download_model_from_hf(self) -> Tuple[str, str]:
        """Download model/metadata from HuggingFace, raising FileNotFoundError on 404."""
        model_filename = f"{self.lang}_{self.gram_size}gram_{self.variant}.parquet"
        metadata_filename = f"{self.lang}_{self.gram_size}gram_{self.variant}_metadata.json"

        def download_from(folder: str) -> Tuple[str, str]:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"models/{folder}/{model_filename}",
                repo_type="model",
            )
            metadata_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"models/{folder}/{metadata_filename}",
                repo_type="model",
            )
            return model_path, metadata_path

        try:
            return download_from(f"{self.variant}_ngram")
        except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError):
            try:
                return download_from("word_ngram")
            except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as fallback_err:
                raise FileNotFoundError(
                    f"Could not load n-gram model for lang='{self.lang}', gram_size={self.gram_size}, "
                    f"variant='{self.variant}' from HuggingFace ({self.repo_id}): {fallback_err}"
                ) from fallback_err
    
    def score(self, text: str) -> float:
        """Score a text using the n-gram model.
        
        Args:
            text (str): Text to score
            
        Returns:
            float: Log probability score of the text
        """
        # Tokenize the text (simple whitespace tokenization)
        tokens = text.strip().split()
        if len(tokens) < self.gram_size:
            return -float('inf')  # Text too short for n-gram model
        
        log_prob = 0.0
        total_ngrams = self.metadata.get('total_ngrams', 1)
        
        # Create n-grams from the text
        for i in range(len(tokens) - self.gram_size + 1):
            ngram = tokens[i:i + self.gram_size]
            
            # Look up the n-gram in the model
            ngram_row = self.model[self.model['ngram'].apply(lambda x: x == ngram)]
            
            if not ngram_row.empty:
                frequency = ngram_row.iloc[0]['frequency']
                # Calculate log probability using frequency
                prob = frequency / total_ngrams
                log_prob += math.log(prob)
            else:
                # Use smoothing for unseen n-grams
                log_prob += math.log(1e-10)  # Small probability for unseen n-grams
        
        return log_prob
    
    def predict_next(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict the next token given a context.
        
        Args:
            context (str): Context to predict next token for
            top_k (int): Number of top predictions to return
            
        Returns:
            List[Tuple[str, float]]: List of (token, probability) pairs
        """
        # Tokenize the context
        context_tokens = context.strip().split()
        
        # For n-gram prediction, we need the last (n-1) tokens as context
        if len(context_tokens) < self.gram_size - 1:
            # Not enough context, return empty list
            return []
        
        # Get the relevant context (last n-1 tokens)
        relevant_context = context_tokens[-(self.gram_size - 1):]
        
        # Find all n-grams that start with this context
        matching_ngrams = []
        for _, row in self.model.iterrows():
            ngram = row['ngram']
            if len(ngram) == self.gram_size and ngram[:-1] == relevant_context:
                next_token = ngram[-1]
                frequency = row['frequency']
                matching_ngrams.append((next_token, frequency))
        
        if not matching_ngrams:
            return []  # No predictions available
        
        # Sort by frequency and convert to probabilities
        matching_ngrams.sort(key=lambda x: x[1], reverse=True)
        total_freq = sum(freq for _, freq in matching_ngrams)
        
        # Convert to probabilities and return top_k
        predictions = []
        for token, freq in matching_ngrams[:top_k]:
            prob = freq / total_freq if total_freq > 0 else 0.0
            predictions.append((token, prob))
        
        return predictions
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size from metadata."""
        if self.metadata and 'unique_ngrams' in self.metadata:
            return self.metadata['unique_ngrams']
        return len(self.model)
    
    @property
    def total_ngrams(self) -> int:
        """Get the total number of n-grams from metadata."""
        if self.metadata and 'total_ngrams' in self.metadata:
            return self.metadata['total_ngrams']
        return len(self.model)
    
    @property
    def size(self) -> int:
        """Get the total number of n-grams in the model."""
        return len(self.model)


def ngram(lang: str, date: str = "latest", gram_size: int = 3, variant: str = "word") -> NGramModel:
    """Create an n-gram model instance.
    
    Args:
        lang (str): Language code (e.g., 'en', 'fr', 'ary')
        date (str): Date of the model (format: YYYYMMDD, default: "latest")
        gram_size (int): Size of n-grams (default: 3)
        variant (str, optional): Model variant (default: "word", can be "subword")
        
    Returns:
        NGramModel: Initialized n-gram model instance
    """
    return NGramModel(lang=lang, date=date, gram_size=gram_size, variant=variant)
