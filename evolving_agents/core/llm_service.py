# evolving_agents/core/llm_service.py

import logging
import os
from typing import List, Dict, Any, Optional

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.backend.errors import ChatModelError

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
        
        # Check if API key is set
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. LLM calls will fail.")
        
        try:
            # Initialize chat model based on provider
            if provider == "openai":
                self.chat_model = OpenAIChatModel(model_id, settings)
            else:
                # Default to OpenAI if provider not supported
                logger.warning(f"Provider {provider} not directly supported, using openai adapter")
                self.chat_model = OpenAIChatModel(model_id, settings)
        except Exception as e:
            logger.error(f"Error initializing LLM service: {str(e)}")
            # Create a placeholder chat model for graceful failure
            self.chat_model = None
    
    async def generate(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated text
        """
        logger.debug(f"Generating text for prompt: {prompt[:100]}...")
        
        if not self.chat_model:
            error_msg = "LLM service not properly initialized. Check API key and configuration."
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
        
        try:
            user_message = UserMessage(prompt)
            response = await self.chat_model.create(messages=[user_message])
            return response.get_text_content()
        except ChatModelError as e:
            error_msg = f"LLM service error: {str(e)}"
            logger.error(error_msg)
            # For testing, return a mock response
            if "OpenAI API key" in str(e) or "API key" in str(e):
                return f"ERROR: OpenAI API key not configured correctly. Please set the OPENAI_API_KEY environment variable."
            return f"ERROR: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error in LLM service: {str(e)}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Mock embedding since we don't have direct access to OpenAI's embedding model in beeai
        logger.debug(f"Generating mock embedding for text: {text[:100]}...")
        
        # Return a simple mock embedding
        return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dimensional vector