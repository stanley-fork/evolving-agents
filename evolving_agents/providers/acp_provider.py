# evolving_agents/providers/acp_provider.py

import logging
from typing import Dict, Any, List, Optional, Union

from evolving_agents.providers.base import FrameworkProvider
from evolving_agents.core.llm_service import LLMService
from evolving_agents.acp.client import ACPClient
from evolving_agents.acp.agent import ACPAgent

logger = logging.getLogger(__name__)

class ACPProvider(FrameworkProvider):
    """
    Provider for BeeAI's Agent Communication Protocol (ACP).
    Handles creation and execution of agents using ACP standards.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None, acp_client: Optional[ACPClient] = None):
        """
        Initialize the ACP provider.
        
        Args:
            llm_service: Optional LLM service to use for agents
            acp_client: Optional pre-configured ACP client
        """
        self.llm_service = llm_service
        self.acp_client = acp_client or ACPClient()
        logger.info("ACP Provider initialized")
    
    async def create_agent(
        self, 
        record: Dict[str, Any],
        tools: Optional[List[Any]] = None,
        firmware_content: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ACPAgent:
        """
        Create an ACP-enabled agent.
        
        Args:
            record: Agent record from the Smart Library
            tools: Optional list of tools to provide to the agent
            firmware_content: Optional firmware content to inject
            config: Optional configuration parameters
            
        Returns:
            Instantiated ACP agent
        """
        logger.info(f"Creating ACP agent '{record['name']}'")
        
        # Apply default config if none provided
        config = config or {}
        
        # Prepare agent description/instructions
        instructions = record["description"]
        if firmware_content:
            instructions = f"{firmware_content}\n\n{instructions}"
        
        # Create ACP agent with the provided specifications
        agent = ACPAgent(
            name=record["name"],
            description=instructions,
            llm_service=self.llm_service,
            tools=tools or [],
            config=config
        )
        
        # Register with ACP client if requested
        if config.get("register_with_acp", False):
            agent.acp_id = await self.acp_client.register_agent(agent)
        
        return agent
    
    async def execute_agent(
        self, 
        agent_instance: Union[ACPAgent, str],
        input_text: str,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an ACP agent with input text.
        
        Args:
            agent_instance: The ACP agent instance or name
            input_text: Input text to process
            execution_config: Optional execution configuration parameters
            
        Returns:
            Execution result dictionary
        """
        logger.info(f"Executing ACP agent with input: {input_text[:50]}...")
        
        # Prepare execution configuration
        execution_config = execution_config or {}
        
        try:
            # Create a text input message following ACP standards
            message = {
                "type": "text_input",
                "content": input_text,
                "message_id": execution_config.get("message_id", "msg_" + str(uuid.uuid4()).replace("-", ""))
            }
            
            # Execute the agent using the ACP client
            response = await self.acp_client.execute_agent(
                agent=agent_instance,
                message=message,
                config=execution_config
            )
            
            # Process and format the response
            result = {
                "status": "success",
                "message": "ACP agent executed successfully",
                "result": self._format_acp_response(response),
                "raw_response": response
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing ACP agent: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "message": f"Error executing ACP agent: {str(e)}",
                "result": f"Error: {str(e)}",
                "error": e
            }
    
    def _format_acp_response(self, response: Any) -> str:
        """Format ACP response into a readable string."""
        if isinstance(response, dict):
            if "content" in response:
                return response["content"]
            return json.dumps(response, indent=2)
        return str(response)
    
    def supports_framework(self, framework_name: str) -> bool:
        """Check if this provider supports the specified framework."""
        return framework_name.lower() in ["acp", "beeai-acp"]
    
    def get_supported_agent_types(self) -> List[str]:
        """Get the agent types supported by this provider."""
        return ["ACPAgent"]
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get the schema for provider configuration."""
        return {
            "transport": {
                "type": "string",
                "enum": ["memory", "http", "stdio", "websocket"],
                "default": "memory",
                "description": "Transport method for ACP communication"
            },
            "message_schema": {
                "type": "string",
                "enum": ["text", "message", "json"],
                "default": "text",
                "description": "Message schema to use for communication"
            },
            "register_with_acp": {
                "type": "boolean", 
                "default": True,
                "description": "Whether to automatically register the agent with the ACP client"
            }
        }