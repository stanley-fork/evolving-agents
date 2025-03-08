# evolving_agents/acp/agent.py

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Callable

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

logger = logging.getLogger(__name__)

class ACPAgent:
    """
    An agent implementation that supports the Agent Communication Protocol (ACP).
    This is a placeholder until BeeAI provides an official implementation.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_service: Any,
        tools: Optional[List[Tool]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an ACP-compatible agent.
        
        Args:
            name: The agent's name
            description: The agent's description
            llm_service: LLM service or chat model to use
            tools: Optional list of tools
            config: Optional configuration
        """
        self.name = name
        self.description = description
        self.config = config or {}
        
        # Get chat model from llm_service
        if hasattr(llm_service, "chat_model"):
            self.chat_model = llm_service.chat_model
        else:
            self.chat_model = llm_service
        
        # Create memory based on config
        memory_type = self.config.get("memory_type", "token")
        if memory_type == "unconstrained":
            self.memory = UnconstrainedMemory()
        else:
            self.memory = TokenMemory(self.chat_model)
        
        # Create meta information
        self.meta = AgentMeta(
            name=name,
            description=description,
            tools=tools or []
        )
        
        # Create internal ReActAgent
        self.agent = ReActAgent(
            llm=self.chat_model,
            tools=tools or [],
            memory=self.memory,
            meta=self.meta
        )
        
        # ACP-specific properties
        self.acp_id = None  # Will be set when registered with ACP client
        self.message_history = []
        
        # Optional callback handlers for specific message types
        self.message_handlers = {}
        
        logger.info(f"Created ACP agent: {name}")
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: The message type to handle
            handler: The handler function
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def process_acp_message(
        self,
        message: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an ACP message.
        
        Args:
            message: The ACP message to process
            config: Optional processing configuration
            
        Returns:
            Response message
        """
        # Record the received message
        self.message_history.append({
            "timestamp": self._get_timestamp(),
            "direction": "received",
            "message": message
        })
        
        # Check for specific message handler
        message_type = message.get("type", "unknown")
        if message_type in self.message_handlers:
            response = await self.message_handlers[message_type](message, config)
            return response
        
        # Default handling based on message type
        if message_type == "text_input":
            # Extract the content and process it
            content = message.get("content", "")
            result = await self.agent.run(content)
            
            # Format the response
            response = {
                "type": "text_output",
                "content": result.result.text if hasattr(result, "result") else str(result),
                "message_id": str(uuid.uuid4()),
                "in_response_to": message.get("message_id")
            }
            
        elif message_type == "message_input":
            # Handle structured message input
            result = await self._process_structured_message(message)
            
            # Format the response
            response = {
                "type": "message_output",
                "content": result,
                "message_id": str(uuid.uuid4()),
                "in_response_to": message.get("message_id")
            }
            
        else:
            # Unknown message type
            response = {
                "type": "error",
                "content": f"Unsupported message type: {message_type}",
                "message_id": str(uuid.uuid4()),
                "in_response_to": message.get("message_id")
            }
        
        # Record the response
        self.message_history.append({
            "timestamp": self._get_timestamp(),
            "direction": "sent",
            "message": response
        })
        
        return response
    
    async def _process_structured_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a structured message input."""
        # This is a placeholder for more sophisticated message handling
        content = message.get("content", {})
        message_type = content.get("type", "unknown")
        
        if message_type == "query":
            # Process a query message
            query = content.get("query", "")
            result = await self.agent.run(query)
            return {
                "type": "query_result",
                "result": result.result.text if hasattr(result, "result") else str(result)
            }
            
        elif message_type == "command":
            # Process a command message
            command = content.get("command", "")
            args = content.get("args", {})
            
            # Simple command processing
            if command == "clear_memory":
                self.memory.clear()
                return {"type": "command_result", "status": "success"}
            elif command == "get_status":
                return {"type": "command_result", "status": "active"}
            else:
                return {
                    "type": "command_result",
                    "status": "error",
                    "message": f"Unknown command: {command}"
                }
        
        else:
            # Default handling
            return {
                "type": "error",
                "message": f"Unsupported structured message type: {message_type}"
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()