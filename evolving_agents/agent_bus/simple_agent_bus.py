# evolving_agents/agent_bus/simple_agent_bus.py

import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleAgentBus:
    """
    A simple implementation of the Agent Bus for demonstration purposes.
    Uses a JSON file to store providers and capabilities.
    """
    def __init__(self, storage_path: str = "agent_bus.json"):
        self.storage_path = storage_path
        self.providers = {}
        self.capabilities = {}
        self._load_data()
    
    def _load_data(self):
        """Load data from storage if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.providers = data.get("providers", {})
                    self.capabilities = data.get("capabilities", {})
                logger.info(f"Loaded Agent Bus data from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading Agent Bus data: {str(e)}")
                self.providers = {}
                self.capabilities = {}
        else:
            logger.info(f"No existing Agent Bus data found at {self.storage_path}, starting fresh.")
    
    def _save_data(self):
        """Save data to storage."""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({
                "providers": self.providers,
                "capabilities": self.capabilities,
                "updated_at": datetime.utcnow().isoformat()
            }, f, indent=2)
        logger.info(f"Saved Agent Bus data to {self.storage_path}")
    
    async def register_provider(
        self, 
        name: str, 
        capabilities: List[Dict[str, Any]], 
        provider_type: str = "AGENT",
        description: str = "",
        metadata: Dict[str, Any] = {}
    ) -> str:
        """Register a provider with its capabilities."""
        # Generate a unique ID for the provider
        provider_id = f"{name.lower().replace(' ', '_')}_{provider_type.lower()}"
        
        # Create the provider record
        self.providers[provider_id] = {
            "id": provider_id,
            "name": name,
            "provider_type": provider_type,
            "description": description,
            "capabilities": capabilities,
            "metadata": metadata,
            "status": "active",
            "registered_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Update capabilities registry
        for capability in capabilities:
            cap_id = capability["id"]
            if cap_id not in self.capabilities:
                self.capabilities[cap_id] = {
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability.get("description", ""),
                    "providers": []
                }
            
            # Add this provider to the capability
            self.capabilities[cap_id]["providers"].append({
                "id": provider_id,
                "name": name,
                "confidence": capability.get("confidence", 0.8)
            })
        
        # Save changes
        self._save_data()
        
        return provider_id
    
    async def update_provider_capabilities(
        self, 
        provider_id: str, 
        capabilities: List[Dict[str, Any]]
    ) -> bool:
        """Update a provider's capabilities."""
        if provider_id not in self.providers:
            return False
        
        # First, remove this provider from all capabilities
        for cap_id in self.capabilities:
            self.capabilities[cap_id]["providers"] = [
                p for p in self.capabilities[cap_id]["providers"] 
                if p["id"] != provider_id
            ]
        
        # Update the provider's capabilities
        self.providers[provider_id]["capabilities"] = capabilities
        self.providers[provider_id]["last_updated"] = datetime.utcnow().isoformat()
        
        # Add the provider to the updated capabilities
        for capability in capabilities:
            cap_id = capability["id"]
            if cap_id not in self.capabilities:
                self.capabilities[cap_id] = {
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability.get("description", ""),
                    "providers": []
                }
            
            # Add this provider to the capability
            self.capabilities[cap_id]["providers"].append({
                "id": provider_id,
                "name": self.providers[provider_id]["name"],
                "confidence": capability.get("confidence", 0.8)
            })
        
        # Save changes
        self._save_data()
        
        return True
    
    async def deregister_provider(self, provider_id: str) -> bool:
        """Deregister a provider."""
        if provider_id not in self.providers:
            return False
        
        # Remove the provider from all capabilities
        for cap_id in self.capabilities:
            self.capabilities[cap_id]["providers"] = [
                p for p in self.capabilities[cap_id]["providers"] 
                if p["id"] != provider_id
            ]
        
        # Remove the provider
        del self.providers[provider_id]
        
        # Save changes
        self._save_data()
        
        return True
    
    async def request_service(
        self, 
        capability: str, 
        content: Dict[str, Any],
        provider_id: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Request a service by capability."""
        # If a specific provider is requested, use it
        if provider_id:
            if provider_id not in self.providers:
                raise ValueError(f"Provider '{provider_id}' not found")
            
            provider = self.providers[provider_id]
            
            # Check if the provider has the requested capability
            has_capability = False
            capability_confidence = 0.0
            for cap in provider["capabilities"]:
                if cap["id"] == capability:
                    confidence = cap.get("confidence", 0.0)
                    if confidence >= min_confidence:
                        has_capability = True
                        capability_confidence = confidence
                        break
            
            if not has_capability:
                raise ValueError(f"Provider '{provider_id}' does not have capability '{capability}' with sufficient confidence")
            
            # For demo purposes, just return a mock response
            return {
                "provider_id": provider_id,
                "provider_name": provider["name"],
                "content": self._mock_process_content(capability, content, provider),
                "confidence": capability_confidence
            }
        
        # Find the best provider for this capability
        if capability not in self.capabilities:
            raise ValueError(f"No providers found for capability '{capability}'")
        
        # Get providers with sufficient confidence
        eligible_providers = [
            p for p in self.capabilities[capability]["providers"]
            if p.get("confidence", 0.0) >= min_confidence
        ]
        
        if not eligible_providers:
            raise ValueError(f"No providers with sufficient confidence for capability '{capability}'")
        
        # Select the provider with the highest confidence
        best_provider = max(eligible_providers, key=lambda p: p.get("confidence", 0.0))
        provider_id = best_provider["id"]
        provider = self.providers[provider_id]
        
        # For demo purposes, just return a mock response
        return {
            "provider_id": provider_id,
            "provider_name": provider["name"],
            "content": self._mock_process_content(capability, content, provider),
            "confidence": best_provider.get("confidence", 0.8)
        }
    
    def _mock_process_content(self, capability: str, content: Dict[str, Any], provider: Dict[str, Any]) -> Dict[str, Any]:
        """Mock processing of content by a provider."""
        # In a real implementation, this would invoke the actual agent/tool
        # For demo purposes, we just return a formatted response
        if "text" in content:
            input_text = content["text"]
        else:
            input_text = json.dumps(content)
            
        return {
            "result": f"Processed by {provider['name']} with capability '{capability}': {input_text[:50]}...",
            "provider_type": provider["provider_type"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "input_size": len(input_text),
                "capability": capability
            }
        }
    
    async def find_providers_for_capability(
        self, 
        capability: str,
        min_confidence: float = 0.5,
        provider_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find providers that offer a specific capability."""
        if capability not in self.capabilities:
            return []
        
        # Get providers with sufficient confidence
        provider_ids = [
            p["id"] for p in self.capabilities[capability]["providers"]
            if p.get("confidence", 0.0) >= min_confidence
        ]
        
        # Filter by provider type if specified
        providers = [
            self.providers[pid] for pid in provider_ids
            if pid in self.providers and (not provider_type or self.providers[pid]["provider_type"] == provider_type)
        ]
        
        return providers
    
    async def find_provider_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a provider by name."""
        for provider_id, provider in self.providers.items():
            if provider["name"] == name:
                return provider
        return None
    
    async def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get a provider by ID."""
        return self.providers.get(provider_id)
    
    async def list_all_providers(self, provider_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered providers."""
        if provider_type:
            return [p for p in self.providers.values() if p["provider_type"] == provider_type]
        return list(self.providers.values())
    
    async def list_all_capabilities(
        self, 
        provider_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List all registered capabilities."""
        result = []
        for cap_id, capability in self.capabilities.items():
            # Filter providers by type and confidence if needed
            filtered_providers = capability["providers"]
            
            if provider_type or min_confidence > 0:
                filtered_providers = []
                for provider in capability["providers"]:
                    provider_record = self.providers.get(provider["id"])
                    if not provider_record:
                        continue
                        
                    if provider_type and provider_record["provider_type"] != provider_type:
                        continue
                        
                    if min_confidence > 0 and provider.get("confidence", 0) < min_confidence:
                        continue
                        
                    filtered_providers.append(provider)
            
            # Skip capabilities with no matching providers
            if not filtered_providers:
                continue
                
            # Add to result
            result.append({
                "id": cap_id,
                "name": capability["name"],
                "description": capability.get("description", ""),
                "providers": filtered_providers
            })
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            result = result[:limit]
            
        return result
    
    async def search_capabilities(
        self, 
        query: str,
        min_confidence: float = 0.5,
        provider_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for capabilities matching a query."""
        # In a real implementation, this would use more sophisticated search
        # For demo purposes, we just do simple text matching
        query = query.lower()
        
        matches = []
        for cap_id, capability in self.capabilities.items():
            # Check if query matches capability name or description
            if query in capability["name"].lower() or query in capability.get("description", "").lower():
                # Filter providers if needed
                filtered_providers = []
                for provider in capability["providers"]:
                    provider_record = self.providers.get(provider["id"])
                    if not provider_record:
                        continue
                        
                    if provider_type and provider_record["provider_type"] != provider_type:
                        continue
                        
                    if min_confidence > 0 and provider.get("confidence", 0) < min_confidence:
                        continue
                        
                    filtered_providers.append(provider)
                
                # Skip capabilities with no matching providers
                if not filtered_providers:
                    continue
                    
                # Add to matches
                matches.append({
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability.get("description", ""),
                    "providers": filtered_providers
                })
        
        # Sort by relevance (for demo, we'll use the number of matching providers)
        matches.sort(key=lambda c: len(c["providers"]), reverse=True)
        
        # Apply limit
        return matches[:limit]