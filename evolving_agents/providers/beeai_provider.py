# evolving_agents/providers/beeai_provider.py (improved version)

import logging
import importlib
import inspect
import json
from typing import Dict, Any, List, Optional, Union
import ast

# BeeAI framework imports
from beeai_framework.agents.react import ReActAgent
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
    ) -> ReActAgent:
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
        
        # Get the ChatModel - prefer provided llm_service
        chat_model = None
        if self.llm_service and self.llm_service.chat_model:
            chat_model = self.llm_service.chat_model
        elif "llm" in config:
            chat_model = config["llm"]
        
        if not chat_model:
            logger.error("No ChatModel available for BeeAgent")
            raise ValueError("ChatModel not available. Provide an LLMService or chat_model in config.")
        
        # Try to determine if this is a class-based agent or a function-based agent
        code_snippet = record["code_snippet"]
        
        # First approach: Look for class with "create_agent" method
        initializer_class_name = None
        try:
            parsed_ast = ast.parse(code_snippet)
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "create_agent":
                            initializer_class_name = node.name
                            break
                    if initializer_class_name:
                        break
        except:
            pass
        
        # If we found an initializer class, use it
        if initializer_class_name:
            try:
                # Create a module from the code snippet
                module_code = f'''
{code_snippet}

# Create a factory function to instantiate the agent
def create_agent_instance(chat_model, tools=None):
    initializer = {initializer_class_name}()
    if hasattr(initializer, "create_agent"):
        # Class method approach
        if "tools" in inspect.signature(initializer.create_agent).parameters:
            return initializer.create_agent(chat_model, tools)
        else:
            return initializer.create_agent(chat_model)
    elif hasattr({initializer_class_name}, "create_agent"):
        # Static method approach
        if "tools" in inspect.signature({initializer_class_name}.create_agent).parameters:
            return {initializer_class_name}.create_agent(chat_model, tools)
        else:
            return {initializer_class_name}.create_agent(chat_model)
    else:
        raise ValueError("No create_agent method found")
'''
                # Create an executable namespace
                namespace = {}
                
                # Execute the code within the namespace
                exec(module_code, namespace)
                
                # Call the factory function
                agent = namespace["create_agent_instance"](chat_model, tools)
                
                # Validate that it's a ReActAgent
                if not isinstance(agent, ReActAgent):
                    raise ValueError(f"Created object is not a ReActAgent: {type(agent)}")
                
                return agent
                
            except Exception as e:
                logger.error(f"Error creating agent using initializer class: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Alternative approach: create a ReActAgent directly
        # Prepare description/instructions
        description = record["description"]
        if firmware_content:
            description = f"{firmware_content}\n\n{description}"
        
        # Create meta information
        meta = AgentMeta(
            name=record["name"],
            description=description,
            tools=tools or []
        )
        
        # Create memory - use TokenMemory by default, but allow configuration
        memory_type = config.get("memory_type", "token")
        if memory_type == "unconstrained":
            memory = UnconstrainedMemory()
        else:  # token memory by default
            memory = TokenMemory(chat_model)
        
        # Create the BeeAgent with proper parameters
        agent = ReActAgent(
            llm=chat_model,
            tools=tools or [],
            memory=memory,
            meta=meta
        )
        
        return agent
    
    async def execute_agent(
        self, 
        agent_instance: Union[ReActAgent, str],
        input_text: str,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a BeeAgent with input text.
        
        Args:
            agent_instance: The BeeAgent instance or name
            input_text: Input text to process
            execution_config: Optional execution configuration parameters
            
        Returns:
            Execution result dictionary with status, message, and result
        """
        if isinstance(agent_instance, str):
            logger.warning(f"Expected agent instance but got name: {agent_instance}. "
                          "This may not work - fixes needed in SystemAgent.execute_item")
            return {
                "status": "error",
                "message": f"BeeAIProvider requires agent instance, not name: {agent_instance}",
                "result": f"Error: BeeAIProvider requires agent instance, not name"
            }
            
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
            # Run the agent with the input text
            run_result = await agent_instance.run(
                prompt=input_text,
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
        return ["BeeAgent", "ReActAgent"]
    
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