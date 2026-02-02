"""
IngestionCoordinator - Task 1.1

Implements an Actor-like pattern using Python asyncio to mimic Swift Actors.
Orchestrates the full ingestion pipeline: Extract → Split → Embed → Index

Requirements:
- CFR-1: Multimodal ingestion (PDF, DOCX, XLSX, XML)
- CFR-2: Semantic chunking (512 tokens, 10% overlap)
- CFR-3: Vector storage (Float16 in SQLite)
- CNFR-1: Offline-first (zero network calls)
- CNFR-2: CPU-only multi-core processing
"""

import asyncio
import time
import tracemalloc
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from extraction_worker import ExtractionWorker
from text_splitter import RecursiveCharacterTextSplitter
from embedding_worker import EmbeddingWorker
from indexing_worker import IndexingWorker


@dataclass
class IngestionResult:
    """Result container for ingestion pipeline."""
    
    file: str
    total_chunks: int = 0
    vectors_stored: int = 0
    processing_time_sec: float = 0.0
    pages_per_sec: float = 0.0
    memory_mb: float = 0.0
    embedding_latency_ms: float = 0.0
    chunk_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'file': self.file,
            'total_chunks': self.total_chunks,
            'vectors_stored': self.vectors_stored,
            'processing_time_sec': round(self.processing_time_sec, 3),
            'pages_per_sec': round(self.pages_per_sec, 2),
            'memory_mb': round(self.memory_mb, 2),
            'embedding_latency_ms': round(self.embedding_latency_ms, 2),
            'chunk_ids': self.chunk_ids,
            'errors': self.errors
        }


class IngestionCoordinator:
    """
    Actor-like coordinator for the document ingestion pipeline.
    
    Uses asyncio for concurrent processing while maintaining
    sequential consistency for shared state (mimics Swift Actor isolation).
    
    Pipeline stages:
    1. Extraction: Parse document to raw text
    2. Splitting: Chunk text into semantic units
    3. Embedding: Generate vector representations
    4. Indexing: Store in vector database
    """
    
    def __init__(
        self,
        output_dir: str = "./output",
        verbose: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 51  # ~10% of 512
    ):
        """
        Initialize the coordinator.
        
        Args:
            output_dir: Directory for output files and database
            verbose: Enable detailed logging
            chunk_size: Target chunk size in tokens (CFR-2: 512)
            chunk_overlap: Overlap between chunks (CFR-2: 10%)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize workers (lazy loading for heavy components)
        self._extraction_worker: Optional[ExtractionWorker] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._embedding_worker: Optional[EmbeddingWorker] = None
        self._indexing_worker: Optional[IndexingWorker] = None
        
        # Actor-like state isolation
        self._lock = asyncio.Lock()
        self._processed_count = 0
    
    @property
    def extraction_worker(self) -> ExtractionWorker:
        """Lazy initialization of extraction worker."""
        if self._extraction_worker is None:
            self._extraction_worker = ExtractionWorker()
        return self._extraction_worker
    
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy initialization of text splitter."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self._token_length
            )
        return self._text_splitter
    
    @property
    def embedding_worker(self) -> EmbeddingWorker:
        """Lazy initialization of embedding worker."""
        if self._embedding_worker is None:
            self._embedding_worker = EmbeddingWorker(
                model_dir=str(self.output_dir.parent / "models" / "onnx")
            )
        return self._embedding_worker
    
    @property
    def indexing_worker(self) -> IndexingWorker:
        """Lazy initialization of indexing worker."""
        if self._indexing_worker is None:
            self._indexing_worker = IndexingWorker(
                db_path=str(self.output_dir / "vectors.db")
            )
        return self._indexing_worker
    
    def _token_length(self, text: str) -> int:
        """
        Count tokens using tiktoken (OpenAI tokenizer).
        Used for accurate chunk size measurement.
        """
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    async def process(self, input_path: str) -> dict:
        """
        Main processing pipeline - orchestrates all workers.
        
        This method implements Actor-like isolation by:
        1. Using async/await for non-blocking I/O
        2. Using asyncio.to_thread for CPU-bound work
        3. Maintaining state consistency with locks
        
        Args:
            input_path: Path to input document
            
        Returns:
            Dictionary with processing results
        """
        # Start memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = IngestionResult(file=input_path)
        path = Path(input_path)
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: EXTRACTION
            # ═══════════════════════════════════════════════════════════
            self._log(f"[1/4] 📄 Extracting text from {path.name}...")
            
            extraction_result = await self.extraction_worker.extract(path)
            
            if extraction_result.get('error'):
                result.errors.append(f"Extraction error: {extraction_result['error']}")
                return result.to_dict()
            
            raw_text = extraction_result['text']
            page_count = extraction_result.get('page_count', 1)
            
            if not raw_text.strip():
                result.errors.append("No text content extracted from document")
                return result.to_dict()
            
            self._log(f"      ✓ Extracted {len(raw_text):,} characters from {page_count} pages")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: CHUNKING
            # ═══════════════════════════════════════════════════════════
            self._log(f"[2/4] ✂️  Splitting into semantic chunks...")
            
            # Run in thread pool to avoid blocking (CPU-bound)
            chunks = await asyncio.to_thread(
                self.text_splitter.split_text,
                raw_text
            )
            
            if not chunks:
                result.errors.append("No chunks created from text")
                return result.to_dict()
            
            self._log(f"      ✓ Created {len(chunks)} chunks (target: {self.chunk_size} tokens, {self.chunk_overlap} overlap)")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: EMBEDDING
            # ═══════════════════════════════════════════════════════════
            self._log(f"[3/4] 🧠 Generating embeddings...")
            
            embeddings = await self.embedding_worker.embed_batch(chunks)
            
            embedding_latency = self.embedding_worker.average_latency_ms
            
            self._log(f"      ✓ Generated {len(embeddings)} embeddings")
            self._log(f"      ✓ Average latency: {embedding_latency:.1f}ms per chunk")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 4: INDEXING
            # ═══════════════════════════════════════════════════════════
            self._log(f"[4/4] 💾 Storing in vector database...")
            
            await self.indexing_worker.initialize()
            
            # Prepare chunk records with metadata
            chunk_records = []
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{path.stem}_{i:04d}"
                
                chunk_record = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'embedding': embedding,
                    'metadata': {
                        'source': str(path.absolute()),
                        'filename': path.name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'token_count': self._token_length(chunk_text),
                        'page_count': page_count
                    }
                }
                chunk_records.append(chunk_record)
            
            stored_count = await self.indexing_worker.store_batch(chunk_records)
            
            self._log(f"      ✓ Stored {stored_count} vectors in database")
            
            # ═══════════════════════════════════════════════════════════
            # FINALIZE RESULTS
            # ═══════════════════════════════════════════════════════════
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result.total_chunks = len(chunks)
            result.vectors_stored = stored_count
            result.processing_time_sec = end_time - start_time
            result.pages_per_sec = page_count / result.processing_time_sec if result.processing_time_sec > 0 else 0
            result.memory_mb = peak / 1024 / 1024
            result.embedding_latency_ms = embedding_latency
            result.chunk_ids = [r['id'] for r in chunk_records]
            
            # Update processed count (actor-like state update)
            async with self._lock:
                self._processed_count += 1
            
        except Exception as e:
            result.errors.append(f"Pipeline error: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            
            # Ensure tracemalloc is stopped
            try:
                tracemalloc.stop()
            except:
                pass
        
        return result.to_dict()
    
    async def process_batch(self, input_paths: List[str]) -> List[dict]:
        """
        Process multiple documents concurrently.
        
        Args:
            input_paths: List of paths to documents
            
        Returns:
            List of result dictionaries
        """
        tasks = [self.process(path) for path in input_paths]
        return await asyncio.gather(*tasks)
    
    @property
    def processed_count(self) -> int:
        """Get count of processed documents."""
        return self._processed_count