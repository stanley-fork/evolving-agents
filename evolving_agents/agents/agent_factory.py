# evolving_agents/agents/agent_factory.py

import logging
import importlib.util
import sys
import os
from typing import Dict, Any, Optional

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import LibraryRecord

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory for creating and executing agents from library records.
    """
    def __init__(self, smart_library: SmartLibrary, llm_service: ChatModel):
        self.library = smart_library
        self.llm = llm_service
        self.active_agents = {}
    
    async def create_agent(
        self, 
        record: LibraryRecord,
        firmware_content: Optional[str] = None
    ) -> Any:
        """
        Create an agent instance from a library record.
        
        Args:
            record: Agent record from the Smart Library
            firmware_content: Optional firmware content to inject
            
        Returns:
            Instantiated agent
        """
        logger.info(f"Creating agent {record.name} from record {record.id}")
        
        # For simplicity, we'll use BeeAgent from beeai_framework
        # In practice, you might want to use the code_snippet to create a custom agent
        
        # Extract tools from record metadata if available
        tools = []
        if record.metadata and "tools" in record.metadata:
            tool_names = record.metadata["tools"]
            for tool_name in tool_names:
                tool_record = await self.library.find_record_by_name(tool_name)
                if tool_record:
                    # Import tool dynamically if needed
                    # For now, we'll just use empty tools list
                    pass
        
        # Create agent with firmware injection
        memory = UnconstrainedMemory()
        
        # In a real implementation, you would process record.code_snippet
        # and create a custom agent based on the code
        # For simplicity, we'll just use BeeAgent
        
        instructions = record.description
        if firmware_content:
            instructions = f"{firmware_content}\n\n{instructions}"
        
        agent = BeeAgent(
            llm=self.llm,
            tools=tools,
            memory=memory,
            meta={
                "name": record.name,
                "description": instructions
            },
            execution=BeeAgentExecutionConfig(max_iterations=10)
        )
        
        # Store in active agents
        self.active_agents[record.id] = agent
        
        return agent
    
    async def execute_agent(self, agent_instance: Any, input_text: str) -> str:
        """
        Execute an agent with input text.
        
        Args:
            agent_instance: The agent instance (typically a BeeAgent)
            input_text: Input text to process
            
        Returns:
            Agent output as string
        """
        logger.info(f"Executing agent with input: {input_text[:50]}...")
        
        output = await agent_instance.run(prompt=input_text)
        return output.result.text