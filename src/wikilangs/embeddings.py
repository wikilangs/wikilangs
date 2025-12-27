"""Embedding module for Wikilangs.

This module provides functionality to load and use word and sentence embeddings
trained on Wikipedia data using BabelVec.
"""

import os
import json
from typing import Optional, Tuple, Union, Dict, Any, List
from pathlib import Path
from huggingface_hub import hf_hub_download


try:
    from babelvec import BabelVec
    BABELVEC_AVAILABLE = True
except ImportError:
    BABELVEC_AVAILABLE = False


class Embeddings:
    """Embeddings for Wikipedia data using BabelVec."""
    
    def __init__(self, lang: str, date: str = "latest", dimension: int = 32):
        """Initialize the embeddings.
        
        Args:
            lang (str): Language code (e.g., 'en', 'fr', 'ary')
            date (str): Date of the model (format: YYYYMMDD, default: "latest")
            dimension (int): Embedding dimension (default: 32)
        
        Raises:
            FileNotFoundError: If the model files cannot be found
            ValueError: If dimension is not supported
        """
        self.date = date
        self.lang = lang
        self.dimension = dimension
        self.resolved_date = date
        self.repo_id = f"wikilangs/{self.lang}"
        
        self.model: Optional[Union[Any, str]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        
        # Load the embeddings
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load the embeddings from HuggingFace."""
        embedding_filename = f"{self.lang}_{self.dimension}d.bin"
        metadata_filename = f"{self.lang}_{self.dimension}d_metadata.json"
        
        embedding_file: Optional[str] = None
        metadata_file: Optional[str] = None
        
        # Load from HuggingFace
        try:
            embedding_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"models/embeddings/monolingual/{embedding_filename}",
                repo_type="model"
            )
            metadata_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=f"models/embeddings/monolingual/{metadata_filename}",
                repo_type="model"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load embeddings for {self.lang} with dimension {self.dimension}: {str(e)}"
            )
        
        if metadata_file is None or embedding_file is None:
            raise FileNotFoundError(f"Could not find embedding files for {self.lang}")

        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load model with BabelVec if available
        if BABELVEC_AVAILABLE:
            import babelvec
            self.model = babelvec.BabelVec.load(embedding_file)
        else:
            self.model = embedding_file  # Return path if BabelVec not available
            
    def embed_word(self, word: str):
        """Get the vector for a specific word."""
        if not BABELVEC_AVAILABLE or self.model is None or isinstance(self.model, str):
            raise ImportError("babelvec is required for embedding operations. Install with: pip install babelvec")
        return self.model.get_word_vector(word)
    
    def embed_sentence(self, sentence: str, method: str = 'average'):
        """Get the vector for a sentence using various methods."""
        if not BABELVEC_AVAILABLE or self.model is None or isinstance(self.model, str):
            raise ImportError("babelvec is required for embedding operations. Install with: pip install babelvec")
        return self.model.get_sentence_vector(sentence, method=method)
    
    @property
    def words(self) -> List[str]:
        """Get the list of words in the vocabulary."""
        if not BABELVEC_AVAILABLE or self.model is None or isinstance(self.model, str):
            return []
        return list(self.model.words)


def embeddings(lang: str, date: str = "latest", dimension: int = 32) -> Union[Embeddings, Tuple[str, Dict]]:
    """Create an embeddings instance.
    
    Args:
        lang (str): Language code (e.g., 'en', 'fr', 'ary')
        date (str): Date of the model (format: YYYYMMDD, default: "latest")
        dimension (int): Embedding dimension (default: 32)
        
    Returns:
        Embeddings: Initialized embeddings instance if babelvec is available,
                   otherwise a tuple of (file_path, metadata)
    """
    emb = Embeddings(lang=lang, date=date, dimension=dimension)
    if BABELVEC_AVAILABLE:
        return emb
    else:
        # If babelvec is not available, emb.model is the file path and emb.metadata is the metadata dict
        model_path = emb.model if isinstance(emb.model, str) else ""
        metadata = emb.metadata if emb.metadata is not None else {}
        return model_path, metadata
