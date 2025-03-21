# evolving_agents/agent_bus/simple_agent_bus.py

import logging
import json
import os
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleAgentBus:
    """
    Enhanced SimpleAgentBus that maintains compatibility with existing code
    while adding the ability to work with actual component instances.
    """
    def __init__(self, storage_path: str = "agent_bus.json"):
        self.storage_path = storage_path
        self.providers = {}  # For backward compatibility
        self.capabilities = {}
        self.components = {}  # New: store actual component instances
        self.llm_service = None
        self._load_data()
    
    def set_llm_service(self, llm_service):
        """Set the LLM service for capability matching."""
        self.llm_service = llm_service
        logger.info("LLM service configured for Agent Bus")
    
    def _load_data(self):
        """Load capability data from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.providers = data.get("providers", {})
                    self.capabilities = data.get("capabilities", {})
                logger.info(f"Loaded capability data from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading capability data: {str(e)}")
                self.providers = {}
                self.capabilities = {}
        else:
            logger.info(f"No existing capability data found at {self.storage_path}")
    
    def _save_data(self):
        """Save capability data to storage."""
        # We don't save component instances, just their registrations
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump({
                "providers": self.providers,
                "capabilities": self.capabilities,
                "updated_at": datetime.utcnow().isoformat()
            }, f, indent=2)
        logger.info(f"Saved capability data to {self.storage_path}")
    
    async def register_provider(
        self, 
        name: str, 
        capabilities: List[Dict[str, Any]], 
        provider_type: str = "AGENT",
        description: str = "",
        metadata: Dict[str, Any] = {},
        instance: Any = None  # New parameter to register actual component instance
    ) -> str:
        """
        Register a provider with its capabilities.
        Optional 'instance' parameter allows registering an actual component instance.
        """
        # Generate a unique ID for the provider
        provider_id = f"{name.lower().replace(' ', '_')}_{provider_type.lower()}"
        
        # Process capabilities
        processed_capabilities = []
        for cap in capabilities:
            if isinstance(cap, str):
                # Convert string capability to dict format
                processed_capabilities.append({
                    "id": cap.lower().replace(" ", "_"),
                    "name": cap,
                    "description": f"Ability to {cap.lower()}",
                    "confidence": 0.8
                })
            elif isinstance(cap, dict):
                # Ensure capability has id and name
                if "id" not in cap:
                    cap["id"] = cap.get("name", "capability").lower().replace(" ", "_")
                if "name" not in cap:
                    cap["name"] = cap["id"].replace("_", " ").title()
                if "confidence" not in cap:
                    cap["confidence"] = 0.8
                processed_capabilities.append(cap)
            else:
                # Skip invalid capabilities
                continue
        
        # Create the provider record
        self.providers[provider_id] = {
            "id": provider_id,
            "name": name,
            "provider_type": provider_type,
            "description": description,
            "capabilities": processed_capabilities,
            "metadata": metadata,
            "status": "active",
            "registered_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            # Add metrics tracking
            "metrics": {
                "total_requests": 0,
                "successful_requests": 0,
                "by_capability": {},
                "by_domain": {}
            }
        }
        
        # Register capabilities
        for capability in processed_capabilities:
            cap_id = capability["id"]
            if cap_id not in self.capabilities:
                self.capabilities[cap_id] = {
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability.get("description", ""),
                    "providers": [],
                    "context": capability.get("context", {})
                }
            
            # Add this provider to the capability
            self.capabilities[cap_id]["providers"].append({
                "id": provider_id,
                "name": name,
                "confidence": capability.get("confidence", 0.8)
            })
        
        # Store component instance if provided (new)
        if instance is not None:
            self.components[provider_id] = {
                "instance": instance,
                "type": provider_type,
                "name": name
            }
            logger.info(f"Registered component instance for '{name}' with ID '{provider_id}'")
        
        self._save_data()
        return provider_id
    
    # Other methods remain compatible, but are enhanced to use component instances when available
    
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
            elif isinstance(cap, dict):
                if "id" not in cap:
                    cap["id"] = cap.get("name", "capability").lower().replace(" ", "_")
                if "name" not in cap:
                    cap["name"] = cap["id"].replace("_", " ").title()
                if "confidence" not in cap:
                    cap["confidence"] = 0.8
                processed_capabilities.append(cap)
            else:
                continue
        
        # Update the provider's capabilities
        self.providers[provider_id]["capabilities"] = processed_capabilities
        self.providers[provider_id]["last_updated"] = datetime.utcnow().isoformat()
        
        # Add the provider to the updated capabilities
        for capability in processed_capabilities:
            cap_id = capability["id"]
            if cap_id not in self.capabilities:
                self.capabilities[cap_id] = {
                    "id": cap_id,
                    "name": capability["name"],
                    "description": capability.get("description", ""),
                    "providers": [],
                    "context": capability.get("context", {})
                }
            
            # Add this provider to the capability
            self.capabilities[cap_id]["providers"].append({
                "id": provider_id,
                "name": self.providers[provider_id]["name"],
                "confidence": capability.get("confidence", 0.8)
            })
        
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
        
        # Remove component instance if it exists
        if provider_id in self.components:
            del self.components[provider_id]
            logger.info(f"Removed component instance for '{provider_id}'")
        
        self._save_data()
        return True
    
    async def request_service(
        self, 
        capability: str, 
        content: Dict[str, Any],
        provider_id: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Request a service by capability.
        Enhanced to use actual component instances when available.
        """
        start_time = datetime.utcnow()
        
        # Extract context and domain if provided
        context = content.pop("_context", None) if isinstance(content, dict) else None
        domain = content.pop("_domain", None) if isinstance(content, dict) else None
        
        try:
            # 1. Find capability
            if capability not in self.capabilities:
                raise ValueError(f"No providers found for capability '{capability}'")
            
            # 2. Get providers with sufficient confidence
            eligible_providers = [
                p for p in self.capabilities[capability]["providers"]
                if p.get("confidence", 0.0) >= min_confidence
            ]
            
            if not eligible_providers:
                raise ValueError(f"No providers with sufficient confidence for capability '{capability}'")
            
            # 3. Select provider
            selected_provider = None
            if provider_id:
                # Find requested provider
                selected_provider = next((p for p in eligible_providers if p["id"] == provider_id), None)
                if not selected_provider:
                    raise ValueError(f"Provider '{provider_id}' not found or doesn't provide capability '{capability}'")
            elif self.llm_service and len(eligible_providers) > 1:
                # Use LLM to select best provider
                selected_provider = await self._select_provider_with_llm(
                    capability, 
                    eligible_providers, 
                    context, 
                    domain
                )
            
            # Default to first provider if needed
            if not selected_provider:
                selected_provider = eligible_providers[0]
            
            # Get provider ID
            provider_id = selected_provider["id"]
            
            # 4. Execute provider's capability
            # Check if we have an actual component instance
            if provider_id in self.components:
                # Use the actual component
                component = self.components[provider_id]
                result = await self._execute_component(
                    component=component,
                    capability=capability,
                    content=content,
                    context=context,
                    domain=domain
                )
                success = True
            else:
                # Fall back to mock processing
                provider = self.providers[provider_id]
                result = self._mock_process_content(capability, content, provider)
                if context:
                    result["metadata"]["context_info"] = f"Context fields: {', '.join(context.keys())}"
                if domain:
                    result["metadata"]["domain"] = domain
                success = True
            
            # 5. Update metrics
            self._update_metrics(
                provider_id=provider_id,
                capability=capability,
                success=success,
                domain=domain
            )
            
            # 6. Return result
            response_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                "provider_id": provider_id,
                "provider_name": selected_provider["name"],
                "content": result,
                "response_time": response_time,
                "confidence": selected_provider.get("confidence", 0.8)
            }
            
        except Exception as e:
            # Update metrics for failure if provider was selected
            if provider_id:
                self._update_metrics(
                    provider_id=provider_id,
                    capability=capability,
                    success=False,
                    domain=domain
                )
            
            # Re-raise the exception
            raise
    
    async def _execute_component(
        self, 
        component: Dict[str, Any],
        capability: str,
        content: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ) -> Any:
        """
        Execute a component with input content.
        
        Args:
            component: Component information
            capability: Capability being used
            content: Input content
            context: Optional context
            domain: Optional domain
            
        Returns:
            Component execution result
        """
        instance = component["instance"]
        component_type = component["type"]
        
        # For BeeAI agents
        if component_type == "AGENT" and hasattr(instance, "run"):
            # If component is a ReActAgent with run method
            if context and isinstance(content, dict):
                # Add context as special field for run
                content_with_context = dict(content)
                content_with_context["_context"] = context
                if domain:
                    content_with_context["_domain"] = domain
                # Convert to prompt if needed
                if "text" in content_with_context:
                    prompt = content_with_context["text"]
                else:
                    prompt = json.dumps(content_with_context)
            else:
                # Use content as-is
                if isinstance(content, dict) and "text" in content:
                    prompt = content["text"]
                else:
                    prompt = str(content)
            
            # Run the agent
            try:
                logger.info(f"Executing agent {component['name']} with prompt: {prompt[:100]}...")
                result = await instance.run(prompt)
                # Extract result text if it's a complex object
                if hasattr(result, "result") and hasattr(result.result, "text"):
                    return result.result.text
                return result
            except Exception as e:
                logger.error(f"Error executing agent {component['name']}: {str(e)}")
                raise
        
        # For BeeAI tools
        elif component_type == "TOOL" and hasattr(instance, "run"):
            # If it's a BeeAI tool with a run method
            try:
                logger.info(f"Executing tool {component['name']}...")
                # Format the input based on tool's needs
                if hasattr(instance, "input_schema"):
                    # Tool expects a specific input schema
                    if isinstance(content, dict):
                        # Try to create an input object from content
                        result = await instance.run(content)
                    else:
                        # Fallback to text
                        result = await instance.run({"text": str(content)})
                else:
                    # Simple text input
                    if isinstance(content, dict) and "text" in content:
                        result = await instance.run(content["text"])
                    else:
                        result = await instance.run(str(content))
                
                # Extract result string if needed
                if hasattr(result, "get_text_content"):
                    return result.get_text_content()
                return result
            except Exception as e:
                logger.error(f"Error executing tool {component['name']}: {str(e)}")
                raise
        
        # For functions or callables
        elif callable(instance):
            # If it's a callable function
            try:
                logger.info(f"Executing function {component['name']}...")
                if asyncio.iscoroutinefunction(instance):
                    # Async function
                    result = await instance(content)
                else:
                    # Sync function
                    result = instance(content)
                return result
            except Exception as e:
                logger.error(f"Error executing function {component['name']}: {str(e)}")
                raise
        
        else:
            # Unsupported component type
            raise ValueError(f"Component {component['name']} has unsupported interface")
    
    def _mock_process_content(self, capability: str, content: Dict[str, Any], provider: Dict[str, Any]) -> Dict[str, Any]:
        """Mock processing of content by a provider."""
        # Legacy method for backward compatibility
        if isinstance(content, dict) and "text" in content:
            input_text = content["text"]
        else:
            input_text = json.dumps(content)
            
        return {
            "result": f"Processed by {provider['name']} with capability '{capability}': {input_text[:50]}...",
            "provider_type": provider["provider_type"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "input_size": len(input_text),
                "capability": capability,
                "note": "This is a mock result. Register component instances for real execution."
            }
        }
    
    async def _select_provider_with_llm(
        self, 
        capability: str, 
        providers: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to select the best provider for a capability.
        
        Args:
            capability: Capability being requested
            providers: List of potential providers
            context: Optional context data
            domain: Optional domain
            
        Returns:
            Selected provider information
        """
        try:
            # Prepare provider information for LLM
            provider_details = []
            for p in providers:
                provider_id = p["id"]
                provider = self.providers.get(provider_id, {})
                metrics = provider.get("metrics", {})
                
                # Get capability-specific metrics
                cap_metrics = metrics.get("by_capability", {}).get(capability, {})
                total_cap_requests = cap_metrics.get("total_requests", 0)
                cap_success_rate = 0
                if total_cap_requests > 0:
                    cap_success_rate = cap_metrics.get("successful_requests", 0) / total_cap_requests
                
                # Get domain-specific metrics if applicable
                domain_success_rate = None
                if domain:
                    domain_metrics = metrics.get("by_domain", {}).get(domain, {})
                    total_domain_requests = domain_metrics.get("total_requests", 0)
                    if total_domain_requests > 0:
                        domain_success_rate = domain_metrics.get("successful_requests", 0) / total_domain_requests
                
                # Add more information for LLM to consider
                has_instance = provider_id in self.components
                
                provider_details.append({
                    "id": provider_id,
                    "name": p["name"],
                    "confidence": p.get("confidence", 0.8),
                    "capability_success_rate": cap_success_rate,
                    "domain_success_rate": domain_success_rate,
                    "total_capability_requests": total_cap_requests,
                    "has_registered_instance": has_instance,
                    "provider_type": provider.get("provider_type", "unknown")
                })
            
            # Create LLM prompt
            prompt = f"""
            Select the best provider for capability '{capability}' based on these criteria:
            
            PROVIDERS:
            {json.dumps(provider_details, indent=2)}
            
            {f"CONTEXT: {json.dumps(context, indent=2)}" if context else ""}
            {f"DOMAIN: {domain}" if domain else ""}
            
            Consider:
            1. Provider confidence for this capability
            2. Success rates (capability-specific and domain-specific if available)
            3. Experience with this capability (number of requests handled)
            4. Whether the provider has a registered instance
            
            Return only the ID of the best provider, nothing else.
            """
            
            # Get LLM recommendation
            response = await self.llm_service.generate(prompt)
            provider_id = response.strip()
            
            # Find the provider in the list
            for p in providers:
                if p["id"] == provider_id or p["name"] == provider_id:
                    return p
            
            # If LLM selection failed, use highest confidence provider
            return max(providers, key=lambda p: p.get("confidence", 0))
            
        except Exception as e:
            logger.warning(f"Error in LLM provider selection: {str(e)}, using highest confidence provider")
            return max(providers, key=lambda p: p.get("confidence", 0))
    
    def _update_metrics(
        self,
        provider_id: str,
        capability: str,
        success: bool,
        domain: Optional[str] = None
    ):
        """
        Update metrics for a provider.
        
        Args:
            provider_id: ID of the provider
            capability: Capability that was used
            success: Whether the execution was successful
            domain: Optional domain context
        """
        if provider_id not in self.providers:
            return
        
        provider = self.providers[provider_id]
        
        # Initialize metrics if needed
        if "metrics" not in provider:
            provider["metrics"] = {
                "total_requests": 0,
                "successful_requests": 0,
                "by_capability": {},
                "by_domain": {}
            }
        
        metrics = provider["metrics"]
        
        # Update overall metrics
        metrics["total_requests"] += 1
        if success:
            metrics["successful_requests"] += 1
        
        # Update capability-specific metrics
        if "by_capability" not in metrics:
            metrics["by_capability"] = {}
        
        if capability not in metrics["by_capability"]:
            metrics["by_capability"][capability] = {
                "total_requests": 0,
                "successful_requests": 0
            }
        
        cap_metrics = metrics["by_capability"][capability]
        cap_metrics["total_requests"] += 1
        if success:
            cap_metrics["successful_requests"] += 1
        
        # Update domain-specific metrics if applicable
        if domain:
            if "by_domain" not in metrics:
                metrics["by_domain"] = {}
            
            if domain not in metrics["by_domain"]:
                metrics["by_domain"][domain] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "by_capability": {}
                }
            
            domain_metrics = metrics["by_domain"][domain]
            domain_metrics["total_requests"] += 1
            if success:
                domain_metrics["successful_requests"] += 1
            
            # Update domain+capability metrics
            if "by_capability" not in domain_metrics:
                domain_metrics["by_capability"] = {}
            
            if capability not in domain_metrics["by_capability"]:
                domain_metrics["by_capability"][capability] = {
                    "total_requests": 0,
                    "successful_requests": 0
                }
            
            dom_cap_metrics = domain_metrics["by_capability"][capability]
            dom_cap_metrics["total_requests"] += 1
            if success:
                dom_cap_metrics["successful_requests"] += 1
        
        # Save changes
        self._save_data()
    
    # ... Other methods with similar enhancements ...
    
    # New methods for component instance registration
    
    async def register_agent(
        self, 
        agent_id: str,
        name: str,
        capabilities: List[Dict[str, Any]],
        agent_instance: Any,
        description: str = "",
        metadata: Dict[str, Any] = {}
    ) -> str:
        """
        Register an actual agent instance with the bus.
        
        Args:
            agent_id: Unique ID for the agent
            name: Human-readable name
            capabilities: List of capabilities provided
            agent_instance: The actual agent instance
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            The agent ID
        """
        return await self.register_provider(
            name=name,
            capabilities=capabilities,
            provider_type="AGENT",
            description=description,
            metadata=metadata,
            instance=agent_instance
        )
    
    async def register_tool(
        self, 
        tool_id: str,
        name: str,
        capabilities: List[Dict[str, Any]],
        tool_instance: Any,
        description: str = "",
        metadata: Dict[str, Any] = {}
    ) -> str:
        """
        Register an actual tool instance with the bus.
        
        Args:
            tool_id: Unique ID for the tool
            name: Human-readable name
            capabilities: List of capabilities provided
            tool_instance: The actual tool instance
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            The tool ID
        """
        return await self.register_provider(
            name=name,
            capabilities=capabilities,
            provider_type="TOOL",
            description=description,
            metadata=metadata,
            instance=tool_instance
        )