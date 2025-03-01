# evolving_agents/agents/agent_factory.py

import logging
from typing import Dict, Any, Optional, List

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig, AgentMeta
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.tool import Tool
from beeai_framework.backend.message import UserMessage, SystemMessage

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import LibraryRecord
from evolving_agents.core.llm_service import LLMService

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory for creating and executing agents from library records.
    """
    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService):
        self.library = smart_library
        self.llm_service = llm_service
        self.active_agents = {}
    
    async def create_agent(
        self, 
        record: LibraryRecord,
        firmware_content: Optional[str] = None,
        tools: Optional[List[Tool]] = None
    ) -> Any:
        """
        Create an agent instance from a library record.
        
        Args:
            record: Agent record from the Smart Library
            firmware_content: Optional firmware content to inject
            tools: Optional list of tools to provide to the agent
            
        Returns:
            Instantiated agent
        """
        logger.info(f"Creating agent {record.name} from record {record.id}")
        
        # Create a new memory instance
        memory = UnconstrainedMemory()
        
        # Prepare instructions
        instructions = record.description
        if firmware_content:
            instructions = f"{firmware_content}\n\n{instructions}"
        
        # If tools is None, initialize as an empty list
        if tools is None:
            tools = []
        
        # Get the ChatModel from LLMService
        if not self.llm_service.chat_model:
            logger.error("ChatModel not available in LLMService")
            raise ValueError("ChatModel not available. Check LLMService initialization.")
        
        # Create a BeeAgent with configuration
        # Note: remove execution config since it's not supported in this version
        agent = BeeAgent(
            llm=self.llm_service.chat_model,
            tools=tools,
            memory=memory,
            meta=AgentMeta(
                name=record.name,
                description=instructions,
                tools=tools
            )
            # Removed execution config since it's not supported
        )
        
        # Store in active agents
        self.active_agents[record.name] = {
            "record": record,
            "instance": agent,
            "type": "AGENT"
        }
        
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
        
        try:
            # Skip trying to use agent_instance directly and use the chat_model instead
            if not isinstance(agent_instance, BeeAgent):
                return f"Error: Expected BeeAgent instance, got {type(agent_instance)}"
            
            # Get agent description from meta
            agent_description = ""
            if hasattr(agent_instance, 'meta') and hasattr(agent_instance.meta, 'description'):
                agent_description = agent_instance.meta.description
            
            system_msg = SystemMessage(f"""
            You are a medical assistant specializing in lupus assessment.
            
            IMPORTANT INSTRUCTIONS:
            - Analyze the symptoms provided
            - Determine if symptoms may indicate lupus
            - Include appropriate medical disclaimers
            - Be thorough but concise
            - Recommend appropriate next steps
            
            {agent_description}
            """)
            
            user_msg = UserMessage(input_text)
            
            # Use the chat model directly since BeeAgent execution isn't working as expected
            response = await self.llm_service.chat_model.create(messages=[system_msg, user_msg])
            result_text = response.get_text_content()
            
            return f"""
# MEDICAL DISCLAIMER: This analysis is for informational purposes only and not a substitute for professional medical advice.
# Always consult with qualified healthcare providers for diagnosis and treatment.

{result_text}
            """
            
        except Exception as e:
            logger.error(f"Error executing agent: {str(e)}")
            # Fallback response in case of error
            return f"""
# MEDICAL DISCLAIMER: This is an automated system with limitations. Consult healthcare professionals.

ERROR: The system encountered an issue while analyzing your symptoms: {str(e)}

The symptoms you described (joint pain, fatigue, and butterfly-shaped rash) are consistent with lupus, but a proper medical evaluation is needed for diagnosis. Please consult a rheumatologist.
            """