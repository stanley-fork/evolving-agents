# evolving_agents/tools/agent_bus/request_service_tool.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json

from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

class RequestInput(BaseModel):
    """Input schema for the RequestServiceTool."""
    capability: str = Field(description="Capability to request")
    content: Union[str, Dict[str, Any]] = Field(description="Content of the request")
    specific_provider: Optional[str] = Field(None, description="Specific provider to request from (optional)")
    min_confidence: float = Field(0.5, description="Minimum confidence level required (0.0-1.0)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional request metadata")

class RequestServiceTool(Tool[RequestInput, None, StringToolOutput]):
    """
    Tool for requesting services from agents through the Agent Bus.
    """
    name = "RequestServiceTool"
    description = "Request services from agents based on their capabilities"
    input_schema = RequestInput
    
    def __init__(
        self, 
        agent_bus,  # We'll use a generic reference to avoid circular imports
        options: Optional[Dict[str, Any]] = None
    ):
        super().__init__(options=options or {})
        self.agent_bus = agent_bus
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "agent_bus", "request"],
            creator=self,
        )
    
    async def _run(self, input: RequestInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Request a service from an agent based on capability.
        
        Args:
            input: Request parameters
        
        Returns:
            StringToolOutput containing the response in JSON format
        """
        try:
            # Prepare content
            if isinstance(input.content, dict):
                content = input.content
            else:
                content = {"text": input.content}
            
            # Add metadata if provided
            if input.metadata:
                content["metadata"] = input.metadata
            
            # Make the request through the Agent Bus
            response = await self.agent_bus.request_service(
                capability=input.capability,
                content=content,
                provider_id=input.specific_provider,
                min_confidence=input.min_confidence
            )
            
            # Return the response
            return StringToolOutput(json.dumps({
                "status": "success",
                "capability": input.capability,
                "provider": response.get("provider_id", "unknown"),
                "provider_name": response.get("provider_name", "Unknown Provider"),
                "content": response.get("content", {}),
                "confidence": response.get("confidence", 0.0)
            }, indent=2))
            
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": f"Error requesting service: {str(e)}",
                "details": traceback.format_exc()
            }, indent=2))
    
    async def request(
        self, 
        capability: str, 
        content: Union[str, Dict[str, Any]],
        specific_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Request a service by capability.
        
        Args:
            capability: Capability to request
            content: Content of the request
            specific_provider: Specific provider to request from (optional)
            
        Returns:
            Response from the provider
        """
        # Prepare content
        if isinstance(content, dict):
            content_dict = content
        else:
            content_dict = {"text": content}
        
        # Make the request
        return await self.agent_bus.request_service(
            capability=capability,
            content=content_dict,
            provider_id=specific_provider
        )
    
    async def call_agent(
        self, 
        agent_name: str, 
        input_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a specific agent directly by name.
        
        Args:
            agent_name: Name of the agent to call
            input_text: Input text for the agent
            metadata: Additional metadata
            
        Returns:
            Response from the agent
        """
        # Prepare content
        content = {"text": input_text}
        if metadata:
            content["metadata"] = metadata
        
        # Find the provider by name
        provider = await self.agent_bus.find_provider_by_name(agent_name)
        if not provider:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        # Make the request to the specific provider
        return await self.agent_bus.request_service(
            capability="process_input",  # Generic capability
            content=content,
            provider_id=provider["id"]
        )
    
    async def send_message(
        self, 
        to_agent: str, 
        message: str,
        from_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message from one agent to another.
        
        Args:
            to_agent: Name of the recipient agent
            message: Message content
            from_agent: Name of the sender agent (optional)
            
        Returns:
            Response from the recipient agent
        """
        # Prepare message content
        content = {
            "message": message,
            "from": from_agent or "SystemAgent"
        }
        
        # Find the recipient by name
        recipient = await self.agent_bus.find_provider_by_name(to_agent)
        if not recipient:
            raise ValueError(f"Agent '{to_agent}' not found")
        
        # Send the message
        return await self.agent_bus.request_service(
            capability="receive_message",  # Standard message receiving capability
            content=content,
            provider_id=recipient["id"]
        )