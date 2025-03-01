# evolving_agents/utils/embeddings.py

import random
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

async def generate_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Generate embeddings for a text string.
    
    Args:
        text: Text to generate embedding for
        model: Model to use for generating embeddings
        
    Returns:
        List of floats representing the embedding
    """
    try:
        # For simplicity, we'll generate a mock embedding
        # In a real implementation, you would call an embedding model API
        logger.debug(f"Generating embedding for text: {text[:100]}...")
        
        # Create a deterministic but simple embedding based on the text content
        # This is NOT suitable for production but works for testing
        random.seed(hash(text) % 10000)
        embedding = [random.random() for _ in range(384)]  # 384-dimensional vector
        
        # Normalize the embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
            
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return a zero vector as fallback
        return [0.0] * 384