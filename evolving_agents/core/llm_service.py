# evolving_agents/core/llm_service.py

import logging
from typing import List, Dict, Any, Optional

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with language models.
    Handles chat and embedding generation.
    """
    def __init__(
        self, 
        provider: str = "openai",
        model_id: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ):
        self.provider = provider
        self.settings = settings or {}
        
        # Initialize chat model based on provider
        if provider == "openai":
            self.chat_model = OpenAIChatModel(model_id, settings)
        else:
            # Default to OpenAI if provider not supported
            logger.warning(f"Provider {provider} not directly supported, using openai adapter")
            self.chat_model = OpenAIChatModel(model_id, settings)
    
    async def generate(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated text
        """
        logger.debug(f"Generating text for prompt: {prompt[:100]}...")
        
        user_message = UserMessage(prompt)
        response = await self.chat_model.create(messages=[user_message])
        
        return response.get_text_content()
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Mock embedding since we don't have direct access to OpenAI's embedding model in beeai
        # In a real implementation, you would use an embedding provider
        logger.debug(f"Generating embedding for text: {text[:100]}...")
        
        # Return a simple mock embedding
        # In reality, you would call an embedding model
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dimensional vector