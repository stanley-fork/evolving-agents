import math
from typing import List, Tuple, Dict, Any, Optional

from evolving_agents.smart_library.record import LibraryRecord

class SearchResult:
    """Result of a semantic search."""
    def __init__(self, record: LibraryRecord, similarity: float):
        self.record = record
        self.similarity = similarity
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record.id,
            "name": self.record.name,
            "similarity": self.similarity
        }

async def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

async def semantic_search(
    query_embedding: List[float],
    records: List[LibraryRecord],
    limit: int = 5,
    threshold: float = 0.0
) -> List[SearchResult]:
    """
    Perform semantic search using embeddings.
    
    Args:
        query_embedding: Embedding vector of the query
        records: List of records to search
        limit: Maximum number of results
        threshold: Minimum similarity threshold
        
    Returns:
        List of search results sorted by similarity
    """
    results = []
    
    for record in records:
        if not record.embedding:
            continue
            
        similarity = await cosine_similarity(query_embedding, record.embedding)
        
        if similarity >= threshold:
            results.append(SearchResult(record, similarity))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x.similarity, reverse=True)
    
    return results[:limit]