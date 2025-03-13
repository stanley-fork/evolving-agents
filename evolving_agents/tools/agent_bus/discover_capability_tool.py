# evolving_agents/tools/agent_bus/discover_capability_tool.py

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
import json

from beeai_framework.tools.tool import Tool, StringToolOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter

class DiscoverInput(BaseModel):
    """Input schema for the DiscoverCapabilityTool."""
    query: Optional[str] = Field(None, description="Search query for capabilities")
    capability_id: Optional[str] = Field(None, description="Specific capability ID to look for")
    provider_type: Optional[str] = Field(None, description="Filter by provider type (AGENT or TOOL)")
    min_confidence: float = Field(0.5, description="Minimum confidence level required (0.0-1.0)")
    limit: int = Field(10, description="Maximum number of results to return")

class DiscoverCapabilityTool(Tool[DiscoverInput, None, StringToolOutput]):
    """
    Tool for discovering capabilities and providers in the Agent Bus.
    """
    name = "DiscoverCapabilityTool"
    description = "Discover available capabilities and providers in the agent ecosystem"
    input_schema = DiscoverInput
    
    def __init__(
        self, 
        agent_bus,  # We'll use a generic reference to avoid circular imports
        options: Optional[Dict[str, Any]] = None
    ):
        super().__init__(options=options or {})
        self.agent_bus = agent_bus
    
    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "agent_bus", "discover"],
            creator=self,
        )
    
    async def _run(self, input: DiscoverInput, options: Optional[Dict[str, Any]] = None, context: Optional[RunContext] = None) -> StringToolOutput:
        """
        Discover capabilities and providers in the Agent Bus.
        
        Args:
            input: Discovery parameters
        
        Returns:
            StringToolOutput containing the discovery results in JSON format
        """
        try:
            if input.capability_id:
                # Find specific capability
                providers = await self.agent_bus.find_providers_for_capability(
                    capability=input.capability_id,
                    min_confidence=input.min_confidence,
                    provider_type=input.provider_type
                )
                
                result = {
                    "status": "success",
                    "capability": input.capability_id,
                    "provider_count": len(providers),
                    "providers": []
                }
                
                for provider in providers:
                    # Find the specific capability details
                    capability_details = None
                    for cap in provider.get("capabilities", []):
                        if cap.get("id") == input.capability_id:
                            capability_details = cap
                            break
                    
                    result["providers"].append({
                        "id": provider["id"],
                        "name": provider["name"],
                        "type": provider.get("provider_type", "AGENT"),
                        "description": provider.get("description", "No description"),
                        "capability_confidence": capability_details.get("confidence", 0.0) if capability_details else 0.0
                    })
                
            elif input.query:
                # Search for capabilities matching the query
                capabilities = await self.agent_bus.search_capabilities(
                    query=input.query,
                    min_confidence=input.min_confidence,
                    provider_type=input.provider_type,
                    limit=input.limit
                )
                
                result = {
                    "status": "success",
                    "query": input.query,
                    "capability_count": len(capabilities),
                    "capabilities": []
                }
                
                for capability in capabilities:
                    result["capabilities"].append({
                        "id": capability["id"],
                        "name": capability["name"],
                        "description": capability.get("description", "No description"),
                        "providers": [
                            {
                                "id": p["id"],
                                "name": p["name"],
                                "confidence": p.get("confidence", 0.0)
                            }
                            for p in capability.get("providers", [])
                        ]
                    })
                
            else:
                # List all capabilities
                all_capabilities = await self.agent_bus.list_all_capabilities(
                    provider_type=input.provider_type,
                    min_confidence=input.min_confidence,
                    limit=input.limit
                )
                
                result = {
                    "status": "success",
                    "capability_count": len(all_capabilities),
                    "capabilities": []
                }
                
                for capability in all_capabilities:
                    result["capabilities"].append({
                        "id": capability["id"],
                        "name": capability["name"],
                        "description": capability.get("description", "No description"),
                        "provider_count": len(capability.get("providers", []))
                    })
            
            return StringToolOutput(json.dumps(result, indent=2))
            
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({
                "status": "error",
                "message": f"Error discovering capabilities: {str(e)}",
                "details": traceback.format_exc()
            }, indent=2))
    
    async def list_providers(self, provider_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered providers.
        
        Args:
            provider_type: Optional filter by provider type
            
        Returns:
            List of providers
        """
        return await self.agent_bus.list_all_providers(provider_type)
    
    async def find_provider(self, capability: str, min_confidence: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Find a provider for a specific capability.
        
        Args:
            capability: Capability to search for
            min_confidence: Minimum confidence level required
            
        Returns:
            Best matching provider or None if not found
        """
        providers = await self.agent_bus.find_providers_for_capability(
            capability=capability,
            min_confidence=min_confidence
        )
        
        if not providers:
            return None
            
        # Return the provider with the highest confidence
        return max(providers, key=lambda p: next(
            (c.get("confidence", 0.0) for c in p.get("capabilities", []) 
             if c.get("id") == capability), 
            0.0
        ))
    
    async def get_capabilities(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get all capabilities for a specific provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of capabilities
        """
        provider = await self.agent_bus.get_provider(provider_id)
        if not provider:
            return []
            
        return provider.get("capabilities", [])
    
    async def search_capabilities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for capabilities matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching capabilities
        """
        return await self.agent_bus.search_capabilities(
            query=query,
            limit=limit
        )