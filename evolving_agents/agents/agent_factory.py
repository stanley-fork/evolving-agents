# evolving_agents/agents/agent_factory.py

import logging
from typing import Dict, Any, Optional, List

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig, AgentMeta, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.tool import Tool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory for creating and executing BeeAgent instances.
    """
    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService):
        self.library = smart_library
        self.llm_service = llm_service
        self.active_agents = {}
    
    async def create_agent(
        self, 
        record: Dict[str, Any],
        firmware_content: Optional[str] = None,
        tools: Optional[List[Tool]] = None
    ) -> BeeAgent:
        """
        Create a BeeAgent instance from a library record.
        
        Args:
            record: Agent record from the Smart Library
            firmware_content: Optional firmware content to inject
            tools: Optional list of tools to provide to the agent
            
        Returns:
            Instantiated BeeAgent
        """
        logger.info(f"Creating agent {record['name']} from record {record['id']}")
        
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
        
        # Create memory
        memory = UnconstrainedMemory()
        
        # If tools is None, initialize as an empty list
        if tools is None:
            tools = []
        
        # Get the ChatModel from LLMService
        if not self.llm_service.chat_model:
            logger.error("ChatModel not available in LLMService")
            raise ValueError("ChatModel not available. Check LLMService initialization.")
        
        # Create the BeeAgent with proper parameters
        agent = BeeAgent(
            llm=self.llm_service.chat_model,
            tools=tools,
            memory=memory,
            meta=meta,
            execution=BeeAgentExecutionConfig(
                max_retries_per_step=3,
                total_max_retries=10,
                max_iterations=20
            )
        )
        
        # Store in active agents
        self.active_agents[record["name"]] = {
            "record": record,
            "instance": agent,
            "type": "AGENT"
        }
        
        return agent
    
    async def execute_agent(self, agent_instance: BeeAgent, input_text: str) -> str:
        """
        Execute a BeeAgent with input text.
        
        Args:
            agent_instance: The BeeAgent instance
            input_text: Input text to process
            
        Returns:
            Agent output as string
        """
        logger.info(f"Executing agent with input: {input_text[:50]}...")
        
        try:
            # Run the agent with the input
            run_result = await agent_instance.run(prompt=input_text)
            
            # Get the text result
            return run_result.result.text
            
        except Exception as e:
            logger.error(f"Error executing agent: {str(e)}")
            return f"Error: {str(e)}"