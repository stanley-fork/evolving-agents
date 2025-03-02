# evolving_agents/core/system_agent.py

import logging
import yaml
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple

from beeai_framework.agents.types import BeeAgentExecutionConfig
from beeai_framework.tools.tool import Tool

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.firmware.firmware import Firmware
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

class SystemAgent:
    """
    Central orchestrator for the evolving agents framework.
    Implements the decision logic from Article 3.1.
    """
    def __init__(
        self, 
        smart_library: SmartLibrary, 
        llm_service: LLMService,
        agent_factory: Optional[AgentFactory] = None,
        tool_factory: Optional[ToolFactory] = None
    ):
        self.library = smart_library
        self.llm = llm_service
        self.firmware = Firmware()
        self.active_items = {}
        
        # Initialize factories
        self.agent_factory = agent_factory or AgentFactory(smart_library, llm_service)
        self.tool_factory = tool_factory or ToolFactory(smart_library, llm_service)
        
        logger.info("System Agent initialized")
    
    async def decide_and_act(
        self, 
        request: str, 
        domain: str,
        record_type: str  # "AGENT" or "TOOL"
    ) -> Dict[str, Any]:
        """
        Main decision logic as described in Article 3.1:
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
        
        # Generate code - avoid f-string when there are curly braces in the template
        creation_prompt = firmware_content + "\n\n"
        creation_prompt += "Create a Python tool that fulfills this request:\n"
        creation_prompt += f"\"{request}\"\n\n"
        creation_prompt += "IMPORTANT REQUIREMENTS:\n"
        creation_prompt += "1. The tool should define a main function that takes a single parameter named 'input' (a string)\n"
        creation_prompt += "2. Set the final result to a variable named 'result'\n"
        creation_prompt += "3. DO NOT include Markdown formatting (```python, etc.)\n"
        creation_prompt += "4. Include appropriate error handling\n"
        creation_prompt += f"5. For domain '{domain}', include all required disclaimers and domain-specific rules\n\n"
        creation_prompt += "EXAMPLE TOOL FUNCTION:\n"
        creation_prompt += "```\n"
        creation_prompt += "# Medical disclaimer here\n\n"
        creation_prompt += "def analyze_symptoms(input):\n"
        creation_prompt += "    # Parse the input\n"
        creation_prompt += "    text = input.lower()\n"
        creation_prompt += "    \n"
        creation_prompt += "    # Process and analyze\n"
        creation_prompt += "    # ...\n"
        creation_prompt += "    \n"
        creation_prompt += "    # Return results\n"
        creation_prompt += "    return {\"result\": \"analysis\", \"confidence\": 0.8}\n\n"
        creation_prompt += "# Call the function with the input text and store the output in 'result'\n"
        creation_prompt += "result = analyze_symptoms(input)\n"
        creation_prompt += "```\n\n"
        creation_prompt += "YOUR CODE (WITHOUT MARKDOWN FORMATTING):\n"

        new_code = await self.llm.generate(creation_prompt)
        
        # Create record
        new_record = await self.library.create_record(
            name=name,
            record_type=record_type,
            domain=domain,
            description=f"Created for: {request}",
            code_snippet=new_code,
            tags=[domain, record_type.lower()]
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
                result = await self.agent_factory.execute_agent(instance, input_text)
            else:  # TOOL
                result = await self.tool_factory.execute_tool(instance, input_text)
            
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