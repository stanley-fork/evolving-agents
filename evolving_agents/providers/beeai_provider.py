# evolving_agents/providers/beeai_provider.py

import logging
from typing import Dict, Any, List, Optional, Union

# BeeAI framework imports
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig, AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.backend.message import UserMessage
from beeai_framework.emitter.emitter import Emitter

from evolving_agents.providers.base import FrameworkProvider
from evolving_agents.core.llm_service import LLMService

logger = logging.getLogger(__name__)

class BeeAIProvider(FrameworkProvider):
    """
    Provider for BeeAI framework integration.
    Handles creation and execution of BeeAgents.
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize the BeeAI provider.
        
        Args:
            llm_service: Optional LLM service to use for agents
        """
        self.llm_service = llm_service
        logger.info("BeeAI Provider initialized")
    
    async def create_agent(
        self, 
        record: Dict[str, Any],
        tools: Optional[List[Tool]] = None,
        firmware_content: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BeeAgent:
        """
        Create a BeeAgent with the specified configuration.
        
        Args:
            record: Agent record from the Smart Library
            tools: Optional list of tools to provide to the agent
            firmware_content: Optional firmware content to inject
            config: Optional configuration parameters
            
        Returns:
            Instantiated BeeAgent
        """
        logger.info(f"Creating BeeAgent '{record['name']}' with {len(tools) if tools else 0} tools")
        
        # Apply default config if none provided
        config = config or {}
        
        # Prepare description/instructions
        instructions = record["description"]
        if firmware_content:
            instructions = f"{firmware_content}\n\n{instructions}"
        
        # Create meta information
        meta = AgentMeta(
            name=record["name"],
            description=instructions,
            tools=tools or []
        )
        
        # Get the ChatModel - prefer provided llm_service, fall back to config
        chat_model = None
        if self.llm_service and self.llm_service.chat_model:
            chat_model = self.llm_service.chat_model
        elif "llm" in config:
            chat_model = config["llm"]
        
        if not chat_model:
            logger.error("No ChatModel available for BeeAgent")
            raise ValueError("ChatModel not available. Provide an LLMService or chat_model in config.")
        
        # Create memory - use TokenMemory by default, but allow configuration
        memory_type = config.get("memory_type", "token")
        if memory_type == "unconstrained":
            memory = UnconstrainedMemory()
        else:  # token memory by default
            memory = TokenMemory(chat_model)
        
        # Create the BeeAgent with proper parameters
        agent = BeeAgent(
            llm=chat_model,
            tools=tools or [],
            memory=memory,
            meta=meta
        )
        
        return agent
    
    async def execute_agent(
        self, 
        agent_instance: BeeAgent,
        input_text: str,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a BeeAgent with input text.
        
        Args:
            agent_instance: The BeeAgent instance
            input_text: Input text to process
            execution_config: Optional execution configuration parameters
            
        Returns:
            Execution result dictionary with status, message, and result
        """
        logger.info(f"Executing BeeAgent with input: {input_text[:50]}...")
        
        # Apply default execution config if none provided
        execution_config = execution_config or {}
        
        # Create execution config with defaults
        bee_exec_config = BeeAgentExecutionConfig(
            max_retries_per_step=execution_config.get("max_retries_per_step", 3),
            total_max_retries=execution_config.get("total_max_retries", 10),
            max_iterations=execution_config.get("max_iterations", 20)
        )
        
        try:
            # Setup observability if requested
            observer = None
            if execution_config.get("enable_observability", False):
                def setup_observer(emitter: Emitter) -> None:
                    def process_events(data: Dict[str, Any], event_meta: Any) -> None:
                        logger.debug(f"BeeAgent event: {event_meta.name} - {str(data)[:100]}")
                    
                    emitter.on("*", process_events)
                
                observer = setup_observer
            
            # Run the agent with the input and config
            run_input = input_text
            
            # Execute with or without observer
            if observer:
                run_result = await agent_instance.run(
                    prompt=run_input,
                    execution=bee_exec_config
                ).observe(observer)
            else:
                run_result = await agent_instance.run(
                    prompt=run_input,
                    execution=bee_exec_config
                )
            
            # Get the text result
            result_text = run_result.result.text
            
            return {
                "status": "success",
                "message": "BeeAgent executed successfully",
                "result": result_text,
                "raw_result": run_result  # Include raw result for advanced usage
            }
            
        except Exception as e:
            logger.error(f"Error executing BeeAgent: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "message": f"Error executing BeeAgent: {str(e)}",
                "result": f"Error: {str(e)}",
                "error": e
            }
    
    def supports_framework(self, framework_name: str) -> bool:
        """
        Check if this provider supports the specified framework.
        
        Args:
            framework_name: Name of the framework to check
            
        Returns:
            True if supported, False otherwise
        """
        return framework_name.lower() in ["beeai", "bee", "bee-framework", "bee_framework"]
    
    def get_supported_agent_types(self) -> List[str]:
        """
        Get the agent types supported by this provider.
        
        Returns:
            List of supported agent type names
        """
        return ["BeeAgent"]
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """
        Get the schema for provider configuration.
        
        Returns:
            Dictionary with configuration options and their schema
        """
        return {
            "memory_type": {
                "type": "string",
                "enum": ["token", "unconstrained"],
                "default": "token",
                "description": "Type of memory to use for the agent"
            },
            "execution": {
                "type": "object",
                "properties": {
                    "max_retries_per_step": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum number of retries per step"
                    },
                    "total_max_retries": {
                        "type": "integer",
                        "default": 10,
                        "description": "Total maximum number of retries"
                    },
                    "max_iterations": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of iterations"
                    }
                }
            },
            "observability": {
                "type": "boolean",
                "default": False,
                "description": "Enable observability for agent execution"
            }
        }