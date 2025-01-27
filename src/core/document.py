# Document dataclass 

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Document:
    """Base document class to store text content and metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None