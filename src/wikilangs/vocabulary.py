"""Vocabulary module for Wikilangs.

This module provides functionality to load and use vocabulary dictionaries
trained on Wikipedia data for different languages and dates.
"""

import os
import json
from typing import Optional, List, Dict, Set
from huggingface_hub import hf_hub_download


class WikilangsVocabulary:
    """Vocabulary dictionary for Wikipedia data.
    
    This class loads pre-trained vocabulary dictionaries from HuggingFace
    based on date and language.
    """
    
    def __init__(self, lang: str, date: str = "latest"):
        """Initialize the vocabulary.
        
        Args:
            lang (str): Language code (e.g., 'en', 'fr', 'ary')
            date (str): Date of the model (format: YYYYMMDD, default: "latest")
        
        Raises:
            FileNotFoundError: If the vocabulary files cannot be found
        """
        self.date = date
        self.lang = lang
        
        # Load the vocabulary
        self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Load the vocabulary from HuggingFace."""
        dict_filename = f"{self.lang}_vocabulary.parquet"
        metadata_filename = f"{self.lang}_vocabulary_metadata.json"
        
        # Load from HuggingFace
        try:
            # Download dictionary file
            dict_file = hf_hub_download(
                repo_id=f"wikilangs/{self.lang}",
                filename=f"models/vocabulary/{dict_filename}",
                repo_type="model"
            )
            import pandas as pd
            # Load parquet file
            self.vocab_df = pd.read_parquet(dict_file)
            
            # Download metadata file
            metadata_file = hf_hub_download(
                repo_id=f"wikilangs/{self.lang}",
                filename=f"models/vocabulary/{metadata_filename}",
                repo_type="model"
            )
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load vocabulary for {self.lang}: {str(e)}"
            )
    
    def lookup(self, word: str) -> Optional[Dict]:
        """Look up a word in the vocabulary.
        
        Args:
            word (str): Word to look up
            
        Returns:
            Optional[Dict]: Word information or None if not found
        """
        matching_rows = self.vocab_df[self.vocab_df['token'] == word]
        if matching_rows.empty:
            return None
        
        row = matching_rows.iloc[0]
        return {
            'token': row['token'],
            'frequency': row['frequency'],
            'idf_score': row['idf_score'],
            'rank': row['rank']
        }
    
    def get_frequency(self, word: str) -> int:
        """Get the frequency of a word in the vocabulary.
        
        Args:
            word (str): Word to get frequency for
            
        Returns:
            int: Frequency of the word
        """
        matching_rows = self.vocab_df[self.vocab_df['token'] == word]
        if matching_rows.empty:
            return 0
        return int(matching_rows.iloc[0]['frequency'])
    
    def get_similar_words(self, word: str, top_k: int = 10) -> List[str]:
        """Get similar words to a given word.
        
        Args:
            word (str): Word to find similar words for
            top_k (int): Number of similar words to return
            
        Returns:
            List[str]: List of similar words
        """
        # For now, return empty list as we don't have similarity data
        # In a real implementation, this would use word embeddings or other similarity measures
        return []
    
    def get_words_with_prefix(self, prefix: str, top_k: int = 10) -> List[str]:
        """Get words that start with a given prefix.
        
        Args:
            prefix (str): Prefix to search for
            top_k (int): Maximum number of words to return
            
        Returns:
            List[str]: List of words with the prefix
        """
        matching_rows = self.vocab_df[self.vocab_df['token'].str.startswith(prefix)]
        matching_rows = matching_rows.sort_values('frequency', ascending=False)
        return matching_rows['token'].head(top_k).tolist()
    
    @property
    def size(self) -> int:
        """Get the vocabulary size."""
        return len(self.vocab_df)
    
    @property
    def words(self) -> Set[str]:
        """Get all words in the vocabulary."""
        return set(self.vocab_df['token'].tolist())
    
    @property
    def vocab_dict(self) -> Dict[str, Dict]:
        """Get vocabulary as a dictionary for backward compatibility."""
        # Convert DataFrame to dict format for backward compatibility
        result = {}
        for _, row in self.vocab_df.iterrows():
            result[row['token']] = {
                'frequency': row['frequency'],
                'idf_score': row['idf_score'],
                'rank': row['rank']
            }
        return result


def vocabulary(lang: str, date: str = "latest") -> WikilangsVocabulary:
    """Create a vocabulary instance.
    
    Args:
        lang (str): Language code (e.g., 'en', 'fr', 'ary')
        date (str): Date of the model (format: YYYYMMDD, default: "latest")
        
    Returns:
        WikilangsVocabulary: Initialized vocabulary instance
    """
    return WikilangsVocabulary(lang=lang, date=date)
