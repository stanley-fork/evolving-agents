# evolving_agents/workflow/workflow_generator.py

import logging
import os
from typing import Dict, List, Any, Optional

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import RecordType

logger = logging.getLogger(__name__)

class WorkflowGenerator:
    """
    Generates workflow YAML from natural language requirements.
    """
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary):
        self.llm = llm_service
        self.library = smart_library
    
    async def generate_workflow(
        self, 
        requirements: str, 
        domain: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a workflow YAML from requirements.
        
        Args:
            requirements: Natural language requirements
            domain: Domain for the workflow
            output_path: Optional path to save the YAML
            
        Returns:
            Generated YAML as string
        """
        logger.info(f"Generating workflow for '{domain}' from requirements: {requirements[:100]}...")
        
        # Get firmware for this domain
        firmware_record = await self.library.get_firmware(domain)
        firmware_content = firmware_record.code_snippet if firmware_record else ""
        
        # Get available records for this domain
        domain_agents = await self.library.find_records_by_domain(domain, RecordType.AGENT)
        domain_tools = await self.library.find_records_by_domain(domain, RecordType.TOOL)
        
        # Format available components for the prompt
        available_components = ""
        if domain_agents:
            available_components += "Available Agents:\n"
            for agent in domain_agents:
                available_components += f"- {agent.name}: {agent.description}\n"
        
        if domain_tools:
            available_components += "\nAvailable Tools:\n"
            for tool in domain_tools:
                available_components += f"- {tool.name}: {tool.description}\n"
        
        # Generate workflow with LLM
        prompt = f"""
        You are an expert workflow generator for an AI agent system.
        
        Generate a YAML workflow based on the following requirements:
        
        REQUIREMENTS:
        {requirements}
        
        DOMAIN: {domain}
        
        {available_components}
        
        The workflow should define the steps needed to accomplish the requirements.
        
        Follow this YAML structure:
        ```yaml
        scenario_name: "A descriptive name"
        domain: "{domain}"
        description: "A brief description of what this workflow does"
        
        # Additional disclaimers for this domain
        additional_disclaimers:
          - "Domain-specific disclaimer 1"
          - "Domain-specific disclaimer 2"
        
        steps:
          - type: "DEFINE"
            item_type: "TOOL or AGENT"
            name: "Name of the item"
            from_existing_snippet: "Name of existing item to reuse/evolve"
            reuse_as_is: true/false
            evolve_changes:
              docstring_update: "Description of changes needed"
            description: "Description of the item's purpose"
            
          - type: "CREATE"
            item_type: "TOOL or AGENT"
            name: "Name of the item"
            
          - type: "EXECUTE"
            item_type: "TOOL or AGENT"
            name: "Name of the item"
            user_input: "Input to provide to the item"
        ```
        
        If the requirements need a new agent, create one with the necessary tools.
        If existing agents or tools can be reused, reference them in the workflow.
        
        GENERATE ONLY THE YAML WITH NO ADDITIONAL TEXT:
        """
        
        workflow_yaml = await self.llm.generate(prompt)
        
        # Clean up the response to extract just the YAML
        workflow_yaml = workflow_yaml.strip()
        
        # Remove markdown code block markers if present
        if workflow_yaml.startswith("```yaml"):
            workflow_yaml = workflow_yaml[7:]
        if workflow_yaml.endswith("```"):
            workflow_yaml = workflow_yaml[:-3]
        
        workflow_yaml = workflow_yaml.strip()
        
        # Save to file if path provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(workflow_yaml)
                logger.info(f"Saved workflow to: {output_path}")
            except Exception as e:
                logger.error(f"Error saving workflow: {str(e)}")
        
        return workflow_yaml