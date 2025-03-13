# evolving_agents/tools/agent_bus/register_provider_tool.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json

from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

class CapabilityModel(BaseModel):
    """Model for a capability."""
    id: str = Field(description="Unique identifier for the capability")
    name: str = Field(description="Human-readable name for the capability")
    description: str = Field(description="Description of what the capability does")
    confidence: float = Field(0.8, description="Confidence level for this capability (0.0-1.0)")

class RegisterInput(BaseModel):
    """Input schema for the RegisterProviderTool."""
    name: str = Field(description="Name of the agent to register")
    agent_type: Optional[str] = Field(None, description="Type of agent (AGENT or TOOL)")
    capabilities: List[Union[str, Dict[str, Any], CapabilityModel]] = Field(
        description="List of capabilities provided by this agent"
    )
    description: Optional[str] = Field(None, description="Description of the agent")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class RegisterProviderTool(Tool[RegisterInput, None, StringToolOutput]):
    """
    Tool for registering agents with the Agent Bus.
    """
    name = "RegisterProviderTool"
    description = "Register agents and their capabilities with the Agent Bus"
    input_schema = RegisterInput
    
    def __init__(
        self, 
        agent_bus,  # We'll use a generic reference to avoid circular imports
        options: Optional[Dict[str, Any]] = None
    ):
        super().__init__(options=options or {})
        self.agent_bus = agent_bus
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "agent_bus", "register"],
            creator=self,
        )
    
    async def _run(self, input: RegisterInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Register an agent with the Agent Bus.
        
        Args:
            input: Registration parameters
        
        Returns:
            StringToolOutput containing the registration result in JSON format
        """
        try:
            # Process capabilities to ensure they have the right format
            processed_capabilities = []
            for cap in input.capabilities:
                if isinstance(cap, str):
                    # Simple string capability - convert to full capability with defaults
                    processed_capabilities.append({
                        "id": cap.lower().replace(" ", "_"),
                        "name": cap,
                        "description": f"Ability to {cap.lower()}",
                        "confidence": 0.8
                    })
                elif isinstance(cap, dict):
                    # Dictionary capability - ensure it has all required fields
                    if "id" not in cap:
                        cap["id"] = cap.get("name", "capability").lower().replace(" ", "_")
                    if "name" not in cap:
                        cap["name"] = cap["id"].replace("_", " ").title()
                    if "description" not in cap:
                        cap["description"] = f"Ability to {cap['name'].lower()}"
                    if "confidence" not in cap:
                        cap["confidence"] = 0.8
                    processed_capabilities.append(cap)
                else:
                    # Assume it's a CapabilityModel
                    processed_capabilities.append(cap.dict())
            
            # Register with the Agent Bus
            provider_id = await self.agent_bus.register_provider(
                name=input.name,
                capabilities=processed_capabilities,
                provider_type=input.agent_type or "AGENT",
                description=input.description or f"Agent providing {len(processed_capabilities)} capabilities",
                metadata=input.metadata or {}
            )
            
            # Return success response
            return StringToolOutput(json.dumps({
                "status": "success",
                "message": f"Successfully registered agent '{input.name}' with {len(processed_capabilities)} capabilities",
                "provider_id": provider_id,
                "registered_capabilities": processed_capabilities
            }, indent=2))
            
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": f"Error registering agent: {str(e)}",
                "details": traceback.format_exc()
            }, indent=2))
    
    async def register(
        self, 
        name: str, 
        capabilities: List[Union[str, Dict[str, Any]]]
    ) -> str:
        """
        Register a provider with its capabilities.
        
        Args:
            name: Name of the agent to register
            capabilities: List of capabilities provided
            
        Returns:
            Provider ID
        """
        # Process capabilities
        processed_capabilities = []
        for cap in capabilities:
            if isinstance(cap, str):
                processed_capabilities.append({
                    "id": cap.lower().replace(" ", "_"),
                    "name": cap,
                    "description": f"Ability to {cap.lower()}",
                    "confidence": 0.8
                })
            else:
                processed_capabilities.append(cap)
        
        # Register with the Agent Bus
        return await self.agent_bus.register_provider(
            name=name,
            capabilities=processed_capabilities
        )
    
    async def update_capabilities(
        self, 
        provider_id: str,
        capabilities: List[Union[str, Dict[str, Any]]]
    ) -> bool:
        """
        Update the capabilities of an existing provider.
        
        Args:
            provider_id: ID of the provider to update
            capabilities: New capabilities list
            
        Returns:
            Success status
        """
        # Process capabilities
        processed_capabilities = []
        for cap in capabilities:
            if isinstance(cap, str):
                processed_capabilities.append({
                    "id": cap.lower().replace(" ", "_"),
                    "name": cap,
                    "description": f"Ability to {cap.lower()}",
                    "confidence": 0.8
                })
            else:
                processed_capabilities.append(cap)
        
        # Update capabilities
        return await self.agent_bus.update_provider_capabilities(
            provider_id=provider_id,
            capabilities=processed_capabilities
        )
    
    async def deregister(self, provider_id: str) -> bool:
        """
        Deregister a provider from the Agent Bus.
        
        Args:
            provider_id: ID of the provider to deregister
            
        Returns:
            Success status
        """
        return await self.agent_bus.deregister_provider(provider_id)