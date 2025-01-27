# Document loading and preprocessing

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataclasses import dataclass
import logging
import PyPDF2
from pathlib import Path
from typing import Dict, Any
from src.core.document import Document 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass

class DocumentProcessor:
    """Handles document preprocessing and loading."""
    
    def convert_pdf_to_text(self, file_path: str) -> str:
        """Convert PDF file to text while preserving basic structure."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error converting PDF to text: {e}")
            raise


    def load_data(self, file_path: str) -> Document:
            """Load document from file and create Document object."""
            try:
                path = Path(file_path)
                if path.suffix.lower() == '.pdf':
                    content = self.convert_pdf_to_text(file_path)
                elif path.suffix.lower() in ['.txt', '.text']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
                
                # Basic metadata
                metadata = {
                    'source': file_path,
                    'file_type': path.suffix.lower(),
                    'file_name': path.name,
                }
                
                return Document(content=content, metadata=metadata)
            except Exception as e:
                logger.error(f"Error loading document: {e}")
                raise

    def add_metadata(self, doc: Document, additional_metadata: Dict[str, Any]) -> Document:
            """Add or update document metadata."""
            doc.metadata.update(additional_metadata)
            return doc

