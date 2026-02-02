"""
RecursiveCharacterTextSplitter - Task 2.1

Implements hierarchical text splitting with:
- 512 token windows (CFR-2)
- 10% overlap (~51 tokens) (CFR-2)
- Separator hierarchy: \n\n → \n → . → space → char

Based on LangChain's implementation but optimized for the PKA pipeline.
"""

from typing import List, Callable, Optional
import re


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a hierarchy of separators.
    
    The algorithm tries to split on the highest-priority separator first,
    then recursively splits any chunks that are still too large using
    progressively finer separators.
    
    This preserves semantic structure by preferring to split on:
    1. Paragraph boundaries (\n\n)
    2. Line breaks (\n)
    3. Sentence endings (. ! ?)
    4. Clause boundaries (, ;)
    5. Word boundaries (space)
    6. Characters (last resort)
    """
    
    # Default separator hierarchy (highest to lowest priority)
    DEFAULT_SEPARATORS = [
        "\n\n",    # Paragraph breaks (strongest boundary)
        "\n",      # Line breaks
        ". ",      # Sentence endings
        "! ",      # Exclamations
        "? ",      # Questions
        "; ",      # Semicolons
        ", ",      # Commas
        " ",       # Word boundaries
        ""         # Character-level (fallback)
    ]
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        separators: Optional[List[str]] = None,
        length_function: Optional[Callable[[str], int]] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Target size in tokens (CFR-2: 512)
            chunk_overlap: Overlap between chunks (CFR-2: ~10% = 51)
            separators: Custom separator hierarchy
            length_function: Function to measure text length (tokens)
            keep_separator: Whether to keep separators in output
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS.copy()
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive separator hierarchy.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks, each approximately chunk_size tokens
        """
        if not text:
            return []
        
        # Start recursive splitting with full separator list
        chunks = self._split_recursive(text, self.separators)
        
        # Post-process: strip whitespace and filter empty
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        chunks = [chunk for chunk in chunks if chunk]
        
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        
        Args:
            text: Text to split
            separators: Remaining separators to try
            
        Returns:
            List of text chunks
        """
        final_chunks = []
        
        # Find the appropriate separator for this text
        separator = separators[-1]  # Default to finest separator
        next_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                # Empty separator = character-level splitting
                separator = sep
                break
            if sep in text:
                separator = sep
                next_separators = separators[i + 1:]
                break
        
        # Split text by chosen separator
        splits = self._split_by_separator(text, separator)
        
        # Merge splits into chunks respecting size limits
        current_chunks: List[str] = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # Case 1: Single split exceeds chunk_size → recurse with finer separator
            if split_length > self.chunk_size:
                # First, finalize any accumulated chunks
                if current_chunks:
                    merged = self._merge_chunks(current_chunks, separator)
                    final_chunks.append(merged)
                    current_chunks = []
                    current_length = 0
                
                # Recurse with finer separators
                if next_separators:
                    sub_chunks = self._split_recursive(split, next_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # No finer separator available, force split
                    forced = self._force_split(split)
                    final_chunks.extend(forced)
            
            # Case 2: Adding split would exceed chunk_size → finalize and start new
            elif current_length + split_length > self.chunk_size:
                if current_chunks:
                    merged = self._merge_chunks(current_chunks, separator)
                    final_chunks.append(merged)
                
                # Start new chunk with overlap from previous
                overlap_chunks = self._get_overlap_chunks(current_chunks, separator)
                current_chunks = overlap_chunks + [split]
                current_length = sum(self.length_function(c) for c in current_chunks)
            
            # Case 3: Split fits → add to current chunk
            else:
                current_chunks.append(split)
                current_length += split_length
        
        # Don't forget the final chunk
        if current_chunks:
            merged = self._merge_chunks(current_chunks, separator)
            final_chunks.append(merged)
        
        return final_chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text by a separator, optionally keeping the separator.
        
        Args:
            text: Text to split
            separator: Separator string
            
        Returns:
            List of splits
        """
        if separator == "":
            # Character-level splitting
            return list(text)
        
        if self.keep_separator:
            # Keep separator at the end of each split (except last)
            parts = text.split(separator)
            result = []
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    # Add separator to end of this part
                    result.append(part + separator)
                elif part:
                    # Last part, no separator
                    result.append(part)
            return result
        else:
            # Simple split, discard separator
            return [s for s in text.split(separator) if s]
    
    def _merge_chunks(self, chunks: List[str], separator: str) -> str:
        """
        Merge chunks back into a single string.
        
        Args:
            chunks: List of text chunks
            separator: Original separator (for non-keep_separator mode)
            
        Returns:
            Merged string
        """
        if self.keep_separator:
            # Separators are already in the chunks
            return ''.join(chunks)
        else:
            return separator.join(chunks)
    
    def _get_overlap_chunks(
        self,
        chunks: List[str],
        separator: str
    ) -> List[str]:
        """
        Get trailing chunks that form the overlap region.
        
        Selects chunks from the end until we reach chunk_overlap tokens.
        This ensures continuity between consecutive chunks.
        
        Args:
            chunks: Previous chunk's splits
            separator: Separator used
            
        Returns:
            List of splits to include in next chunk's start
        """
        if not chunks or self.chunk_overlap <= 0:
            return []
        
        overlap_chunks = []
        overlap_length = 0
        
        # Work backwards through chunks
        for chunk in reversed(chunks):
            chunk_length = self.length_function(chunk)
            
            if overlap_length + chunk_length <= self.chunk_overlap:
                overlap_chunks.insert(0, chunk)
                overlap_length += chunk_length
            else:
                # Would exceed overlap limit
                break
        
        return overlap_chunks
    
    def _force_split(self, text: str) -> List[str]:
        """
        Force split text into chunk_size pieces when no separator works.
        
        Tries to break at word boundaries where possible.
        
        Args:
            text: Text that's too long even at character level
            
        Returns:
            List of forced chunks
        """
        chunks = []
        
        # Use word boundaries where possible
        words = text.split(' ')
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = self.length_function(word)
            
            # If single word exceeds chunk_size, split the word
            if word_length > self.chunk_size:
                # Finalize current chunk first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the long word by characters
                for i in range(0, len(word), self.chunk_size):
                    chunks.append(word[i:i + self.chunk_size])
            
            elif current_length + word_length + 1 > self.chunk_size:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - 2)  # Keep last 2 words
                current_chunk = current_chunk[overlap_start:] + [word]
                current_length = sum(self.length_function(w) for w in current_chunk) + len(current_chunk) - 1
            
            else:
                current_chunk.append(word)
                current_length += word_length + (1 if current_chunk else 0)  # +1 for space
        
        # Final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def split_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Split multiple texts and return as document objects.
        
        Args:
            texts: List of texts to split
            metadatas: Optional metadata for each text
            
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        metadatas = metadatas or [{}] * len(texts)
        
        for text, metadata in zip(texts, metadatas):
            chunks = self.split_text(text)
            
            for i, chunk in enumerate(chunks):
                doc = {
                    'content': chunk,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': self.length_function(chunk)
                    }
                }
                documents.append(doc)
        
        return documents