# evolving_agents/core/system_agent.py (improved version)

import logging
import yaml
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig, AgentMeta
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.tools.tool import Tool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider

logger = logging.getLogger(__name__)

class SystemAgent:
    """
    Central orchestrator for the evolving agents framework.
    Implements the decision logic:
    - If similarity >= 0.8: Reuse
    - If 0.4 <= similarity < 0.8: Evolve
    - If similarity < 0.4: Create new
    """
    def __init__(
    self, 
    smart_library: SmartLibrary, 
    llm_service: LLMService,
    agent_factory: Optional[AgentFactory] = None,
    tool_factory: Optional[ToolFactory] = None,
    provider_registry: Optional[ProviderRegistry] = None
    ):
        self.library = smart_library
        self.llm = llm_service
        self.firmware = Firmware()
        self.active_items = {}
        
        # Initialize provider registry
        self.provider_registry = provider_registry or ProviderRegistry()
        if not self.provider_registry.list_available_providers():
            self.provider_registry.register_provider(BeeAIProvider(llm_service))
        
        # Initialize factories
        self.agent_factory = agent_factory or AgentFactory(
            smart_library, 
            llm_service,
            self.provider_registry
        )
        self.tool_factory = tool_factory or ToolFactory(smart_library, llm_service)
        
        # Set up templates
        from evolving_agents.utils.setup_templates import setup_templates
        setup_templates()
        
        logger.info("System Agent initialized")
    
    async def decide_and_act(
        self, 
        request: str, 
        domain: str,
        record_type: str  # "AGENT" or "TOOL"
    ) -> Dict[str, Any]:
        """
        Main decision logic:
        - If similarity >= 0.8: Reuse
        - If 0.4 <= similarity < 0.8: Evolve
        - If similarity < 0.4: Create new
        
        Args:
            request: User request text
            domain: Domain for the request
            record_type: Type of record to find/create
            
        Returns:
            Result of the action
        """
        logger.info(f"Processing {record_type} request for domain '{domain}': {request[:50]}...")
        
        # Get firmware for this domain
        firmware_content = self.firmware.get_firmware_prompt(domain)
        
        # Search for semantically similar records
        search_results = await self.library.semantic_search(
            query=request,
            record_type=record_type,
            domain=domain,
            limit=1
        )
        
        # Decision logic based on similarity
        if search_results and search_results[0][1] >= 0.8:
            # High similarity: Reuse existing
            record, similarity = search_results[0]
            result = await self._reuse_existing(record, similarity, request)
        elif search_results and search_results[0][1] >= 0.4:
            # Medium similarity: Evolve existing
            record, similarity = search_results[0]
            result = await self._evolve_existing(record, similarity, request, domain, firmware_content)
        else:
            # Low similarity: Create new
            result = await self._create_new(request, domain, record_type, firmware_content)
        
        return result
    
    async def _reuse_existing(
        self,
        record: Dict[str, Any],
        similarity: float,
        request: str
    ) -> Dict[str, Any]:
        """
        Reuse an existing record as-is.
        
        Args:
            record: The record to reuse
            similarity: Similarity score
            request: Original request
            
        Returns:
            Result dictionary
        """
        logger.info(f"Reusing existing {record['record_type']} '{record['name']}' (similarity={similarity:.2f})")
        
        # Update usage metrics
        await self.library.update_usage_metrics(record["id"], True)
        
        # Create instance based on record type
        if record["record_type"] == "AGENT":
            instance = await self.agent_factory.create_agent(record)
        else:  # TOOL
            instance = await self.tool_factory.create_tool(record)
        
        # Add to active items
        self.active_items[record["name"]] = {
            "record": record,
            "instance": instance,
            "type": record["record_type"]
        }
        
        return {
            "status": "success",
            "action": "reuse",
            "record": record,
            "similarity": similarity,
            "message": f"Reused existing {record['record_type']} '{record['name']}' (similarity={similarity:.2f})"
        }
    
    async def _evolve_existing(
        self,
        record: Dict[str, Any],
        similarity: float,
        request: str,
        domain: str,
        firmware_content: str
    ) -> Dict[str, Any]:
        """
        Evolve an existing record with minimal changes.
        
        Args:
            record: Record to evolve
            similarity: Similarity score
            request: Original request
            domain: Domain of the request
            firmware_content: Firmware to inject
            
        Returns:
            Result dictionary
        """
        logger.info(f"Evolving {record['record_type']} '{record['name']}' (similarity={similarity:.2f})")
        
        # Generate evolved code
        evolution_prompt = f"""
        {firmware_content}

        We have an existing {record['record_type'].lower()}:
        \"\"\"
        {record['code_snippet']}
        \"\"\"

        The user request is:
        \"{request}\"

        Evolve or refactor the existing snippet to better address the user request,
        while preserving the original's core logic. Ensure all firmware rules are met.
        
        For domain '{domain}', make sure to include all required disclaimers and domain-specific rules.
        
        For AGENT records: The agent should use BeeAI's ReActAgent and properly use tools.
        For TOOL records: The tool should extend beeai_framework.tools.tool.Tool class.
        
        EVOLVED CODE:
        """

        new_code = await self.llm.generate(evolution_prompt)
        
        # Create evolved record
        evolved_record = await self.library.evolve_record(
            parent_id=record["id"],
            new_code_snippet=new_code,
            description=f"Evolved for: {request}"
        )
        
        # Create instance based on record type
        if evolved_record["record_type"] == "AGENT":
            instance = await self.agent_factory.create_agent(evolved_record, firmware_content)
        else:  # TOOL
            instance = await self.tool_factory.create_tool(evolved_record, firmware_content)
        
        # Add to active items
        self.active_items[evolved_record["name"]] = {
            "record": evolved_record,
            "instance": instance,
            "type": evolved_record["record_type"]
        }
        
        return {
            "status": "success",
            "action": "evolve",
            "record": evolved_record,
            "from_record": record,
            "similarity": similarity,
            "message": f"Evolved {record['record_type']} '{record['name']}' to version {evolved_record['version']} (similarity={similarity:.2f})"
        }
    
    
    async def _create_new(
    self,
    request: str,
    domain: str,
    record_type: str,
    firmware_content: str
    ) -> Dict[str, Any]:
        """
        Create a new record from scratch.
        
        Args:
            request: User request
            domain: Domain of the request
            record_type: Type of record to create
            firmware_content: Firmware to inject
            
        Returns:
            Result dictionary
        """
        # Generate a name based on the request
        words = request.split()
        name = f"{domain.capitalize()}_{record_type}_{words[0]}" if words else f"{domain.capitalize()}_{record_type}"
        
        logger.info(f"Creating new {record_type} '{name}' for request: {request[:50]}...")
        
        # Get paths to template files
        import os
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        
        # Make sure the templates directory exists
        os.makedirs(templates_dir, exist_ok=True)
        
        agent_template_path = os.path.join(templates_dir, "beeai_agent_template.txt")
        tool_template_path = os.path.join(templates_dir, "beeai_tool_template.txt")
        
        # Read templates from files or use default if files don't exist
        try:
            with open(agent_template_path, 'r') as f:
                bee_agent_template = f.read()
        except FileNotFoundError:
            logger.warning(f"Template file {agent_template_path} not found, using default template")
            bee_agent_template = "# Default BeeAI agent template would be here"
            
        try:
            with open(tool_template_path, 'r') as f:
                bee_tool_template = f.read()
        except FileNotFoundError:
            logger.warning(f"Template file {tool_template_path} not found, using default template")
            bee_tool_template = "# Default BeeAI tool template would be here"
        
        # Generate code - different approach for agents vs tools
        if record_type == "AGENT":
            creation_prompt = f"""
            {firmware_content}

            Create a Python agent using the BeeAI framework that fulfills this request:
            "{request}"

            IMPORTANT REQUIREMENTS:
            1. The agent must be a properly implemented BeeAI ReActAgent 
            2. Use the following framework imports:
            - from beeai_framework.agents.react import ReActAgent
            - from beeai_framework.agents.types import AgentMeta
            - from beeai_framework.memory import TokenMemory or UnconstrainedMemory

            3. The agent must follow this structure - implementing a class with a create_agent method:

            REFERENCE TEMPLATE FOR A BEEAI AGENT:
            ```python
    {bee_agent_template}
            ```

            YOUR TASK:
            Create a similar agent class for: "{request}"
            - Replace the WeatherAgentInitializer with an appropriate name for this domain
            - Adapt the description and functionality for the {domain} domain
            - Include all required disclaimers from the firmware
            - Specify any tools the agent should use
            - The code must be complete and executable

            CODE:
            """
        else:  # TOOL
            creation_prompt = f"""
            {firmware_content}

            Create a Python tool using the BeeAI framework that fulfills this request:
            "{request}"

            IMPORTANT REQUIREMENTS:
            1. The tool must be a properly implemented BeeAI Tool class
            2. Use the following framework imports:
            - from beeai_framework.tools.tool import Tool, StringToolOutput
            - from beeai_framework.context import RunContext
            - from pydantic import BaseModel, Field

            3. The tool must follow this structure with an input schema and _run method:

            REFERENCE TEMPLATE FOR A BEEAI TOOL:
            ```python
    {bee_tool_template}
            ```

            YOUR TASK:
            Create a similar tool class for: "{request}"
            - Replace WeatherTool with an appropriate name for this domain
            - Replace WeatherToolInput with an appropriate input schema
            - Define proper input fields with descriptions
            - Implement the _run method with appropriate logic
            - Include error handling
            - For domain '{domain}', include all required disclaimers
            - The code must be complete and executable

            CODE:
            """

        new_code = await self.llm.generate(creation_prompt)
        
        # Create record
        metadata = {"framework": "beeai"}
        
        # Add metadata for required_tools if we detect it in the code
        if record_type == "AGENT":
            # Extract required tools from the code
            import re
            tools_pattern = r"tools=\[(.*?)\]"
            tools_match = re.search(tools_pattern, new_code, re.DOTALL)
            
            if tools_match:
                tools_str = tools_match.group(1)
                # Try to extract tool class names
                tool_classes = re.findall(r'(\w+)\(\)', tools_str)
                if tool_classes:
                    metadata["required_tools"] = tool_classes
        
        new_record = await self.library.create_record(
            name=name,
            record_type=record_type,
            domain=domain,
            description=f"Created for: {request}",
            code_snippet=new_code,
            tags=[domain, record_type.lower()],
            metadata=metadata
        )
        
        # Create instance based on record type
        if record_type == "AGENT":
            instance = await self.agent_factory.create_agent(new_record, firmware_content)
        else:  # TOOL
            instance = await self.tool_factory.create_tool(new_record, firmware_content)
        
        # Add to active items
        self.active_items[new_record["name"]] = {
            "record": new_record,
            "instance": instance,
            "type": record_type
        }
        
        return {
            "status": "success",
            "action": "create",
            "record": new_record,
            "message": f"Created new {record_type} '{name}' for request: {request[:50]}..."
        }
    
    async def execute_item(self, name: str, input_text: str) -> Dict[str, Any]:
        """
        Execute an active item (agent or tool).
        
        Args:
            name: Name of the item to execute
            input_text: Input text for the item
            
        Returns:
            Execution result
        """
        if name not in self.active_items:
            return {
                "status": "error",
                "message": f"Item '{name}' not found in active items"
            }
        
        item = self.active_items[name]
        record = item["record"]
        instance = item["instance"]
        
        logger.info(f"Executing {record['record_type']} '{name}' with input: {input_text[:50]}...")
        
        try:
            # Execute based on record type
            if record["record_type"] == "AGENT":
                result = await self.agent_factory.execute_agent(
                    name,  # Use name instead of instance
                    input_text
                )
            else:  # TOOL
                result = await self.tool_factory.execute_tool(
                    instance, 
                    input_text
                )
            
            # Update usage metrics
            await self.library.update_usage_metrics(record["id"], True)
            
            return {
                "status": "success",
                "item_name": name,
                "item_type": record["record_type"],
                "result": result,
                "message": f"Executed {record['record_type']} '{name}'"
            }
        except Exception as e:
            logger.error(f"Error executing {record['record_type']} '{name}': {str(e)}")
            
            # Update usage metrics as failure
            await self.library.update_usage_metrics(record["id"], False)
            
            return {
                "status": "error",
                "message": f"Error executing {record['record_type']} '{name}': {str(e)}"
            }