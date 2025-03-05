# evolving_agents/utils/embeddings.py

import logging
import os
import numpy as np
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and comparing text embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            api_key: Optional OpenAI API key. If not provided, will look for OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Embeddings will fall back to mock implementation.")
        
        # Cache to avoid redundant embedding API calls
        self.embedding_cache = {}
    
    async def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Generate embeddings for a text string using OpenAI's API.
        
        Args:
            text: Text to generate embedding for
            model: Model to use for generating embeddings
            
        Returns:
            List of floats representing the embedding
        """
        # Check cache first
        cache_key = f"{model}:{text}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            if not self.api_key:
                # Fall back to mock implementation
                return self._generate_mock_embedding(text)
            
            # Use OpenAI's API for embeddings
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Ensure text isn't too long (truncate if needed)
            max_length = 8000
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            response = client.embeddings.create(
                input=truncated_text,
                model=model
            )
            
            # Extract the embedding
            embedding = response.data[0].embedding
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a mock embedding as fallback
            return self._generate_mock_embedding(text)
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate a mock embedding for testing purposes.
        
        Args:
            text: Text to generate mock embedding for
            
        Returns:
            A deterministic but simple embedding based on the text content
        """
        import hashlib
        import random
        
        # Create a hash of the text for deterministic randomness
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        random.seed(text_hash)
        
        # Generate a 1536-dimensional vector (matching OpenAI's typical embedding size)
        embedding = [random.random() for _ in range(1536)]
        
        # Normalize the embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
            
        return embedding
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0-1 where 1 is identical)
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0
            
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure the result is between 0 and 1
        return max(0, min(1, similarity))

# Initialize a global instance for use throughout the application
embedding_service = EmbeddingService()