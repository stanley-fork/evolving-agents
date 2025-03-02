# evolving_agents/core/llm_service.py

import logging
import os
from typing import Dict, Any, List, Optional

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName

logger = logging.getLogger(__name__)

class OpenAIChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "openai"

    def __init__(self, model_id: str | None = None, settings: dict | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}
        
        super().__init__(
            model_id if model_id else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
            provider_id="openai",
            settings=_settings,
        )

class LLMService:
    """
    LLM service that interfaces with BeeAI's ChatModel.
    """
    def __init__(
        self, 
        provider: str = "openai", 
        api_key: Optional[str] = None, 
        model: str = None
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Save API key to environment if provided
        if self.api_key and provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Initialize ChatModel
        if provider == "openai":
            self.chat_model = OpenAIChatModel(
                model_id=model or "gpt-4o",
                settings={"api_key": self.api_key} if self.api_key else None
            )
        elif provider == "ollama":
            # Use LiteLLM for Ollama too
            self.chat_model = LiteLLMChatModel(
                model_id=model or "llama3",
                provider_id="ollama",
                settings={}
            )
        else:
            # Default to OpenAI
            self.chat_model = OpenAIChatModel(
                model_id=model or "gpt-4o",
                settings={"api_key": self.api_key} if self.api_key else None
            )
        
        logger.info(f"Initialized LLM service with provider: {provider}, model: {self.chat_model.model_id}")
    
    async def generate(self, prompt: str) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated text
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # Create a new message
        message = UserMessage(prompt)
        
        try:
            # Generate response
            response = await self.chat_model.create(messages=[message])
            return response.get_text_content()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return a fallback response in case of error
            return f"Error generating response: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        # In future, use BeeAI's embedding capabilities if/when available
        # For now, return a simple placeholder vector
        return [0.1, 0.2, 0.3, 0.4, 0.5]