import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any, Tuple
import uuid
import logging
from src.core.document import Document
from ..connector import DBConnector, RetrievedChunk

logger = logging.getLogger(__name__)

class ChromaConnector(DBConnector):
    """ChromaDB implementation of database connector."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.client = None
        self.collections = {}

    def connect(self) -> None:
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Connected to ChromaDB")
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {e}")
            raise

    def disconnect(self) -> None:
        self.client = None
        self.collections = {}
        logger.info("Disconnected from ChromaDB")

    def create_collection(self, collection_name: str) -> None:
        if not self.client:
            raise ConnectionError("Not connected to ChromaDB")
        
        try:
            existing_collections = self.client.list_collections()
            if collection_name in existing_collections:
                logger.info(f"Collection {collection_name} already exists")
                return
            self.collections[collection_name] = self.client.create_collection(
                name=collection_name
            )
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def drop_collection(self, collection_name: str) -> None:
        if not self.client:
            raise ConnectionError("Not connected to ChromaDB")
        
        try:
            self.client.delete_collection(collection_name)
            self.collections.pop(collection_name, None)
            logger.info(f"Dropped collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error dropping collection {collection_name}: {e}")
            raise

    def list_collections(self) -> List[str]:
        if not self.client:
            raise ConnectionError("Not connected to ChromaDB")
        
        try:
            collections = self.client.list_collections()
            logger.info(f"Found {len(collections)} collections")
            return collections
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise

    def get_collection(self, collection_name: str):
        """Helper method to get or create collection reference."""
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name
            )
        return self.collections[collection_name]

    def add_to_collection(
        self, 
        collection_name: str, 
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
        ) -> List[str]:

        if not self.client:
            raise ConnectionError("Not connected to ChromaDB")
        
        try:
            collection = self._get_collection(collection_name)
            
            # Generate IDs for chunks
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]

            # Store chunks with their embeddings
            collection.add(
                embeddings=embeddings,
                documents=chunks,      # Store original text chunks
                metadatas=metadata,
                ids=chunk_ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to collection {collection_name}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding chunks to collection {collection_name}: {e}")
            raise

    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        k: int = 5
        ) -> List[Tuple[str, float, Dict[str, Any]]]:

        if not self.client:
            raise ConnectionError("Not connected to ChromaDB")
        
        try:
            collection = self._get_collection(collection_name)
            expected_dim = self._collection_dimensions[collection_name]
            
            # Validate query embedding dimension
            if len(query_embedding) != expected_dim:
                raise ValueError(
                    f"Query embedding has dimension {len(query_embedding)}, "
                    f"expected {expected_dim}"
                )
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['distances', 'metadatas']
            )
            
            # Format results as (id, similarity_score, metadata) tuples
            formatted_results = [
                (id_, 1.0 - dist, meta)  # Convert distance to similarity score
                for id_, dist, meta in zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )
            ]
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching in collection {collection_name}: {e}")
            raise