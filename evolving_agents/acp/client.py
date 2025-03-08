# evolving_agents/acp/client.py

import logging
import json
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class ACPClient:
    """
    A basic implementation of an ACP client that manages agent communication.
    This is a placeholder until BeeAI provides an official implementation.
    """
    
    def __init__(self, transport: str = "memory"):
        """
        Initialize the ACP client.
        
        Args:
            transport: The transport mechanism to use. Options:
                - "memory": In-memory transport (default, for testing)
                - "http": HTTP with Server-Sent Events
                - "stdio": Standard input/output
                - "websocket": WebSocket connection
        """
        self.transport = transport
        self.agent_registry = {}  # Maps agent IDs to agent instances
        self.message_history = []  # Stores message history for debugging
        logger.info(f"Initialized ACP client with {transport} transport")
    
    async def register_agent(self, agent_instance: Any) -> str:
        """
        Register an agent with the ACP service.
        
        Args:
            agent_instance: The agent instance to register
            
        Returns:
            A unique identifier for the agent
        """
        agent_id = str(uuid.uuid4())
        self.agent_registry[agent_id] = agent_instance
        logger.info(f"Registered agent with ACP: {agent_id}")
        return agent_id
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the ACP service.
        
        Args:
            agent_id: The agent ID to unregister
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            logger.info(f"Unregistered agent from ACP: {agent_id}")
            return True
        logger.warning(f"Attempted to unregister unknown agent: {agent_id}")
        return False
    
    async def execute_agent(
        self, 
        agent: Union[str, Any],
        message: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an agent with a message.
        
        Args:
            agent: The agent ID or instance
            message: The message to send to the agent
            config: Optional configuration for execution
            
        Returns:
            The agent's response
        """
        # Resolve agent instance
        agent_instance = None
        if isinstance(agent, str):
            if agent in self.agent_registry:
                agent_instance = self.agent_registry[agent]
            else:
                raise ValueError(f"Unknown agent ID: {agent}")
        else:
            agent_instance = agent
        
        # Format message following ACP standards
        acp_message = self._format_acp_message(message)
        
        # Simulate agent execution
        if hasattr(agent_instance, "process_acp_message"):
            # If agent has ACP-specific handler
            response = await agent_instance.process_acp_message(acp_message, config)
        elif hasattr(agent_instance, "run"):
            # Use standard run method if available
            if "content" in acp_message:
                run_result = await agent_instance.run(acp_message["content"])
                response = {
                    "type": "text_output",
                    "content": run_result.result.text if hasattr(run_result, "result") else str(run_result),
                    "message_id": str(uuid.uuid4()),
                    "in_response_to": acp_message.get("message_id")
                }
            else:
                raise ValueError("Message content not found")
        else:
            raise ValueError(f"Agent does not support execution")
        
        # Record the message exchange
        self.message_history.append({
            "timestamp": self._get_timestamp(),
            "direction": "request",
            "message": acp_message
        })
        self.message_history.append({
            "timestamp": self._get_timestamp(),
            "direction": "response",
            "message": response
        })
        
        return response
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Send a message from one agent to another.
        
        Args:
            sender_id: The ID of the sending agent
            recipient_id: The ID of the receiving agent
            message: The message content (string or dict)
            
        Returns:
            The recipient agent's response
        """
        # Validate agent IDs
        if sender_id not in self.agent_registry:
            raise ValueError(f"Unknown sender agent ID: {sender_id}")
        if recipient_id not in self.agent_registry:
            raise ValueError(f"Unknown recipient agent ID: {recipient_id}")
        
        # Format the message
        if isinstance(message, str):
            acp_message = {
                "type": "text_input",
                "content": message,
                "message_id": str(uuid.uuid4()),
                "sender_id": sender_id
            }
        else:
            acp_message = message
            if "message_id" not in acp_message:
                acp_message["message_id"] = str(uuid.uuid4())
            acp_message["sender_id"] = sender_id
        
        # Get the recipient agent
        recipient = self.agent_registry[recipient_id]
        
        # Deliver the message and get response
        response = await self.execute_agent(recipient, acp_message)
        
        # Add response metadata
        response["recipient_id"] = recipient_id
        
        return response
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get the message exchange history."""
        return self.message_history
    
    def _format_acp_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Format a message according to ACP standards."""
        # Add required ACP fields if not present
        if "message_id" not in message:
            message["message_id"] = str(uuid.uuid4())
        
        # Determine message type if not specified
        if "type" not in message:
            if "content" in message and isinstance(message["content"], str):
                message["type"] = "text_input"
            else:
                message["type"] = "message_input"
        
        return message
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    async def _transport_send(self, message: Dict[str, Any]) -> None:
        """Send a message using the configured transport."""
        if self.transport == "memory":
            # In-memory transport doesn't need to do anything
            pass
        elif self.transport == "http":
            # Placeholder for HTTP implementation
            logger.info(f"HTTP transport: {json.dumps(message)}")
        elif self.transport == "stdio":
            # Write to stdout
            print(json.dumps(message))
        elif self.transport == "websocket":
            # Placeholder for WebSocket implementation
            logger.info(f"WebSocket transport: {json.dumps(message)}")
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")