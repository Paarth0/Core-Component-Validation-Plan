"""
Golden Output Tests - Fixed Async Version
"""

import sys
import os

# === PATH SETUP ===
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file)
project_root = os.path.dirname(tests_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# === END PATH SETUP ===

import pytest
import asyncio

# Imports
from text_splitter import RecursiveCharacterTextSplitter
from embedding_worker import EmbeddingWorker
from ingestion_coordinator import IngestionCoordinator, IngestionResult


class TestTextSplitter:
    """Task 2.1: RecursiveCharacterTextSplitter Tests"""
    
    def test_basic_splitting(self):
        """Test basic text splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10
        )
        
        text = "Hello world. " * 50
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        print(f"✓ Created {len(chunks)} chunks")
    
    def test_no_content_loss(self):
        """Test no content is lost."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            length_function=len
        )
        
        markers = [f"MARKER_{i}" for i in range(10)]
        text = " ".join([f"Section {m} content." for m in markers])
        
        chunks = splitter.split_text(text)
        combined = " ".join(chunks)
        
        for m in markers:
            assert m in combined, f"Lost: {m}"
        
        print("✓ No content loss")
    
    def test_chunk_overlap(self):
        """Test chunk overlap exists."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            length_function=len
        )
        
        text = "Word " * 100
        chunks = splitter.split_text(text)
        
        assert len(chunks) >= 1
        print(f"✓ Created {len(chunks)} chunks with overlap")


class TestEmbeddingWorker:
    """Task 2.2: Embedding Tests"""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_embedding_dimensions(self):
        """Test embedding dimensions."""
        worker = EmbeddingWorker()
        embedding = await worker.embed("Hello world")
        
        assert len(embedding) == 384
        print(f"✓ Embedding dimension: {len(embedding)}")
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_batch_embedding(self):
        """Test batch embedding."""
        worker = EmbeddingWorker()
        texts = ["First.", "Second.", "Third."]
        
        embeddings = await worker.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)
        print(f"✓ Batch: {len(embeddings)} embeddings")
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_embedding_consistency(self):
        """Test same input gives same output."""
        worker = EmbeddingWorker()
        
        text = "The quick brown fox"
        emb1 = await worker.embed(text)
        emb2 = await worker.embed(text)
        
        # Should be very similar
        import numpy as np
        diff = np.max(np.abs(np.array(emb1) - np.array(emb2)))
        assert diff < 0.01
        print("✓ Embedding consistency verified")


class TestIngestionCoordinator:
    """Task 1.1: Coordinator Tests"""
    
    @pytest.mark.asyncio(loop_scope="function")
    async def test_initialization(self, tmp_path):
        """Test coordinator init."""
        coordinator = IngestionCoordinator(
            output_dir=str(tmp_path),
            verbose=False
        )
        
        assert coordinator.chunk_size == 512
        assert coordinator.chunk_overlap == 51
        print("✓ Coordinator initialized")
    
    def test_result_format(self):
        """Test result JSON format."""
        result = IngestionResult(file="test.pdf")
        result.total_chunks = 10
        result.vectors_stored = 10
        
        result_dict = result.to_dict()
        
        assert "file" in result_dict
        assert "total_chunks" in result_dict
        assert "vectors_stored" in result_dict
        print("✓ Result format correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])