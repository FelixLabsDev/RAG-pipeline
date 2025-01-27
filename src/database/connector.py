from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from src.core.document import Document

logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """Class to store retrieved chunk with its metadata and similarity score."""
    text: str                    # Original text chunk
    metadata: Dict[str, Any]     # Metadata about the chunk
    similarity_score: float      # Similarity score from vector search
    doc_id: str                  # ID of the source document

class DBConnector(ABC):
    """Abstract base class for database connections."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    def drop_collection(self, collection_name: str) -> None:
        """Drop an existing collection."""
        pass

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all available collections."""
        pass

    @abstractmethod
    def add_to_collection(
        self, 
        collection_name: str, 
        chunks: List[str],           
        embeddings: List[List[float]], 
        metadata: List[Dict[str, Any]] 
    ) -> List[str]:  
        """Add text chunks and their embeddings to a specific collection."""
        pass


    @abstractmethod
    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar embeddings.
        Returns:
            List of tuples containing (id, similarity_score, metadata)
        """
        pass