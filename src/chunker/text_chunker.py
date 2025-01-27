from typing import List, Dict, Any, Optional
import logging
from langchain.text_splitter import CharacterTextSplitter
from core.document import Document

logger = logging.getLogger(__name__)

class TextChunker:
    """Class for handling text chunking operations."""
    
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
        separator: str = "\n\n",
        length_function: callable = len,
        add_start_index: bool = True
    ):
        """
        Initialize text chunker with specified parameters.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: String separator to split on
            length_function: Function to measure text length
            add_start_index: Whether to add chunk start index to metadata
        """
        self.text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )
        self.add_start_index = add_start_index
        logger.info(
            f"Initialized TextChunker with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def create_chunks(
        self, 
        document: Document,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Split document into chunks while preserving metadata.
        
        Args:
            document: Document object to chunk
            chunk_size: Optional override for chunk size
            chunk_overlap: Optional override for chunk overlap
            
        Returns:
            List of Document objects representing chunks
        """
        try:
            # Update splitter parameters if provided
            if chunk_size is not None:
                self.text_splitter.chunk_size = chunk_size
            if chunk_overlap is not None:
                self.text_splitter.chunk_overlap = chunk_overlap
            
            # Split the text
            chunk_texts = self.text_splitter.split_text(document.content)
            
            # Create new documents for chunks
            chunked_docs = []
            for i, chunk_text in enumerate(chunk_texts):
                # Create new metadata dict for the chunk
                chunk_metadata = document.metadata.copy()
                
                # Add chunk-specific metadata
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                })
                
                # Add start index if requested
                if self.add_start_index:
                    chunk_metadata["chunk_start_index"] = document.content.find(chunk_text)
                
                # Create new Document for the chunk
                chunk_doc = Document(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    doc_id=f"{document.doc_id}_chunk_{i}" if document.doc_id else None
                )
                chunked_docs.append(chunk_doc)
            
            logger.info(f"Split document into {len(chunked_docs)} chunks")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
