"""
EmbeddingWorker - Task 2.2

Uses transformers library directly (no sentence-transformers dependency).
This avoids all the version compatibility issues.

Requirements:
- AC: Average embedding latency < 50ms per chunk on CPU
- CFR-3: Output Float16 format for storage efficiency
"""

import asyncio
import time
import numpy as np
from typing import List
from pathlib import Path


class EmbeddingWorker:
    """
    Generates embeddings using transformers library directly.
    
    Uses all-MiniLM-L6-v2 model which produces 384-dimensional embeddings.
    """
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize the embedding worker.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._tokenizer = None
        
        # Performance tracking
        self._total_embeddings = 0
        self._total_time_ms = 0.0
    
    def _load_model(self):
        """Load the model using transformers."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Run: pip install transformers torch"
            )
        
        print(f"Loading model: {self.MODEL_NAME}...")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME)
        self._model.eval()  # Set to evaluation mode
        
        print("Model loaded!")
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]
    
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """
        # Load model (lazy initialization)
        await asyncio.to_thread(self._load_model)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            start_time = time.perf_counter()
            
            # Run embedding in thread pool
            embeddings = await asyncio.to_thread(
                self._encode_batch, batch
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._total_embeddings += len(batch)
            self._total_time_ms += elapsed_ms
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of texts using transformers directly.
        """
        import torch
        
        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**encoded)
        
        # Mean pooling - take mean of token embeddings
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded['attention_mask']
        
        # Expand attention mask for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings (only non-masked tokens)
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean
        embeddings = sum_embeddings / sum_mask
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy and then Float16
        embeddings_np = embeddings.numpy().astype(np.float16)
        
        return embeddings_np.tolist()
    
    @property
    def average_latency_ms(self) -> float:
        """Get average embedding latency in milliseconds."""
        if self._total_embeddings == 0:
            return 0.0
        return self._total_time_ms / self._total_embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding vector dimension."""
        return self.EMBEDDING_DIM
    
    @property
    def total_embeddings(self) -> int:
        """Get total number of embeddings generated."""
        return self._total_embeddings
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._total_embeddings = 0
        self._total_time_ms = 0.0