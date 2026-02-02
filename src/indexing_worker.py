"""
IndexingWorker - Vector Storage

Stores embeddings in SQLite database with Float16 format (CFR-3).
Uses WAL mode for concurrent read/write safety.
"""

import asyncio
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class IndexingWorker:
    """
    Manages vector storage using SQLite.
    
    Features:
    - Float16 embedding storage (CFR-3)
    - WAL mode for concurrent access (R-3)
    - Brute-force cosine similarity search
    """
    
    def __init__(self, db_path: str = "./output/vectors.db"):
        """
        Initialize the indexing worker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection and schema."""
        if self._initialized:
            return
        
        await asyncio.to_thread(self._init_db)
        self._initialized = True
    
    def _init_db(self):
        """Internal database initialization."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # WAL mode for concurrent access
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=10000")
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT DEFAULT 'float16',
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            )
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_created 
            ON chunks(created_at)
        """)
        
        self._conn.commit()
    
    async def store_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Store multiple chunks with their embeddings.
        
        Args:
            chunks: List of {id, text, embedding, metadata}
            
        Returns:
            Number of chunks stored
        """
        return await asyncio.to_thread(self._store_batch_sync, chunks)
    
    def _store_batch_sync(self, chunks: List[Dict[str, Any]]) -> int:
        """Synchronous batch storage."""
        stored = 0
        
        for chunk in chunks:
            try:
                chunk_id = chunk['id']
                text = chunk['text']
                embedding = chunk['embedding']
                metadata = chunk.get('metadata', {})
                
                # Store chunk
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunks (id, text, metadata) VALUES (?, ?, ?)",
                    (chunk_id, text, json.dumps(metadata))
                )
                
                # Convert to Float16 (CFR-3)
                embedding_array = np.array(embedding, dtype=np.float16)
                embedding_blob = embedding_array.tobytes()
                
                # Store vector
                self._conn.execute(
                    "INSERT OR REPLACE INTO vectors (chunk_id, embedding, dimension, dtype) VALUES (?, ?, ?, ?)",
                    (chunk_id, embedding_blob, len(embedding), 'float16')
                )
                
                stored += 1
                
            except Exception as e:
                print(f"Error storing chunk {chunk.get('id')}: {e}")
        
        self._conn.commit()
        return stored
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of matching chunks with scores
        """
        return await asyncio.to_thread(
            self._search_sync, query_embedding, top_k
        )
    
    def _search_sync(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Brute-force cosine similarity search."""
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            return []
        
        query = query / query_norm
        
        cursor = self._conn.execute(
            "SELECT chunk_id, embedding FROM vectors"
        )
        
        similarities = []
        for row in cursor:
            stored = np.frombuffer(
                row['embedding'], dtype=np.float16
            ).astype(np.float32)
            
            stored_norm = np.linalg.norm(stored)
            if stored_norm > 0:
                stored = stored / stored_norm
                similarity = float(np.dot(query, stored))
                similarities.append((row['chunk_id'], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in similarities[:top_k]:
            chunk_row = self._conn.execute(
                "SELECT text, metadata FROM chunks WHERE id = ?",
                (chunk_id,)
            ).fetchone()
            
            if chunk_row:
                results.append({
                    'id': chunk_id,
                    'text': chunk_row['text'],
                    'score': score,
                    'metadata': json.loads(chunk_row['metadata']) if chunk_row['metadata'] else {}
                })
        
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await asyncio.to_thread(self._get_stats_sync)
    
    def _get_stats_sync(self) -> Dict[str, Any]:
        """Synchronous stats retrieval."""
        chunk_count = self._conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        
        vector_count = self._conn.execute(
            "SELECT COUNT(*) FROM vectors"
        ).fetchone()[0]
        
        return {
            'chunk_count': chunk_count,
            'vector_count': vector_count,
            'db_path': self.db_path
        }
    
    async def delete(self, chunk_id: str) -> bool:
        """Delete a chunk and its vector."""
        return await asyncio.to_thread(self._delete_sync, chunk_id)
    
    def _delete_sync(self, chunk_id: str) -> bool:
        """Synchronous delete."""
        self._conn.execute("DELETE FROM vectors WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        self._conn.commit()
        return True
    
    async def clear(self):
        """Clear all data from database."""
        await asyncio.to_thread(self._clear_sync)
    
    def _clear_sync(self):
        """Synchronous clear."""
        self._conn.execute("DELETE FROM vectors")
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False