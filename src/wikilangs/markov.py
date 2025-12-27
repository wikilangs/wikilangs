"""Markov chain module for Wikilangs.

This module provides functionality to load and use Markov chain models
trained on Wikipedia data for different languages and dates.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError


class MarkovChain:
    """Markov chain model for text generation."""
    
    def __init__(self, lang: str, date: str = "latest", depth: int = 2, variant: str = "word"):
        """Initialize the Markov chain model.
        
        Args:
            lang (str): Language code (e.g., 'en', 'fr', 'ary')
            date (str): Date of the model (format: YYYYMMDD, default: "latest")
            depth (int): Depth of the Markov chain (default: 2)
            variant (str, optional): Model variant (default: "word", can be "subword")
        
        Raises:
            FileNotFoundError: If the model files cannot be found
            ValueError: If depth is not supported
        """
        self.date = date
        self.lang = lang
        self.depth = depth
        self.resolved_date = date
        self.variant = variant  # "word" or "subword"
        self.repo_id = f"wikilangs/{self.lang}"
        
        self.transitions = None
        self.metadata = None
        
        # Validate depth
        supported_depths = [1, 2, 3, 4, 5]
        if depth not in supported_depths:
            raise ValueError(f"Unsupported depth: {depth}. "
                           f"Supported depths: {supported_depths}")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Markov chain model from HuggingFace."""
        model_file, metadata_file = self._download_model_from_hf()
        
        self.transitions = pd.read_parquet(model_file)
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

    def _download_model_from_hf(self) -> Tuple[str, str]:
        """Download model/metadata from HuggingFace, raising FileNotFoundError on 404."""
        model_filename = f"{self.lang}_markov_ctx{self.depth}_{self.variant}.parquet"
        metadata_filename = f"{self.lang}_markov_ctx{self.depth}_{self.variant}_metadata.json"

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
            return download_from(f"{self.variant}_markov")
        except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as err:
            try:
                return download_from("word_markov")
            except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as fallback_err:
                raise FileNotFoundError(
                    f"Could not load Markov chain model for lang='{self.lang}', depth={self.depth}, "
                    f"date={self.date} from HuggingFace ({self.repo_id}): {fallback_err}"
                ) from fallback_err
    
    def generate(self, length: int = 100, seed: Optional[List[str]] = None) -> str:
        """Generate text using the Markov chain model.
        
        Args:
            length (int): Length of text to generate (in tokens)
            seed (List[str], optional): Seed tokens to start generation
            
        Returns:
            str: Generated text
        """
        if seed is None:
            # Start with a random context from the model
            random_row = self.transitions.sample(n=1).iloc[0]
            raw_context = random_row['context']
            if isinstance(raw_context, str):
                current_context = json.loads(raw_context)
            else:
                current_context = list(raw_context)
        else:
            if len(seed) < self.depth:
                # Pad seed if too short
                current_context = seed + ['<pad>'] * (self.depth - len(seed))
            else:
                current_context = seed[-self.depth:]
        
        generated_tokens = current_context.copy()
        max_retries = 10  # Prevent infinite loops
        
        for i in range(length):
            # Get transitions for current context
            transitions = self.get_transitions(tuple(current_context))
            
            retry_count = 0
            while not transitions and retry_count < max_retries:
                # No transitions available, try to find a random context
                if len(self.transitions) > 0:
                    random_row = self.transitions.sample(n=1).iloc[0]
                    raw_context = random_row['context']
                    if isinstance(raw_context, str):
                        current_context = json.loads(raw_context)
                    else:
                        current_context = list(raw_context)
                    transitions = self.get_transitions(tuple(current_context))
                    retry_count += 1
                else:
                    break
            
            if not transitions:
                # Still no transitions after retries, stop generation
                break
            
            # Sample next token based on probabilities
            tokens = list(transitions.keys())
            probabilities = list(transitions.values())
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
                try:
                    next_token = random.choices(tokens, weights=probabilities)[0]
                except (IndexError, ValueError):
                    next_token = random.choice(tokens) if tokens else None
            else:
                next_token = random.choice(tokens) if tokens else None
            
            if next_token is None:
                break
                
            generated_tokens.append(next_token)
            
            # Update context for next iteration
            current_context = current_context[1:] + [next_token]
        
        # Clean up tokens and join
        cleaned_tokens = []
        for token in generated_tokens:
            if token.startswith('▁'):
                cleaned_tokens.append(token[1:])  # Remove SentencePiece prefix
            else:
                cleaned_tokens.append(token)
        
        return ' '.join(cleaned_tokens)
    
    def get_transitions(self, state: Tuple[str, ...]) -> Dict[str, float]:
        """Get transition probabilities for a given state.
        
        Args:
            state (Tuple[str, ...]): Current state (tuple of tokens)
            
        Returns:
            Dict[str, float]: Dictionary of next tokens and their probabilities
        """
        # Convert state to list for comparison
        state_list = list(state)
        
        # Find all transitions that match this context
        def context_matches(context_data):
            if isinstance(context_data, str):
                try:
                    parsed_context = json.loads(context_data)
                    return parsed_context == state_list
                except:
                    return False
            else:
                return list(context_data) == state_list
        
        matching_transitions = self.transitions[
            self.transitions['context'].apply(context_matches)
        ]
        
        if matching_transitions.empty:
            return {}
        
        # Create dictionary of next tokens and their probabilities
        transitions_dict = {}
        for _, row in matching_transitions.iterrows():
            next_token = row['next_token']
            probability = row['probability']
            transitions_dict[next_token] = probability
        
        return transitions_dict
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.metadata.get("vocab_size", 0)
    
    @property
    def supported_depths(self) -> List[int]:
        """Get list of supported depths."""
        return [1, 2, 3, 4, 5]
    
    @property
    def size(self) -> int:
        """Get the total number of transitions in the model."""
        return len(self.transitions)
    
    @property
    def total_transitions(self) -> int:
        """Get the total number of transitions in the model."""
        return len(self.transitions)


def markov(lang: str, date: str = "latest", depth: int = 2, variant: str = "word") -> MarkovChain:
    """Factory function to create a MarkovChain instance.
    
    Args:
        lang (str): Language code (e.g., 'en', 'fr', 'ary')
        date (str): Date string (e.g., '20251201', default: "latest")
        depth (int): Depth of the Markov chain (default: 2)
        variant (str): Model variant (default: "word", can be "subword")
        
    Returns:
        MarkovChain: Initialized Markov chain model
    """
    return MarkovChain(lang=lang, date=date, depth=depth, variant=variant)
