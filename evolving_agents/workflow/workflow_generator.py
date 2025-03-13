# evolving_agents/workflow/workflow_generator.py

import logging
import os
from typing import Dict, Any, Optional, Union
import re

# Import the interface instead of the concrete class
from evolving_agents.core.base import IAgent
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary

logger = logging.getLogger(__name__)

class WorkflowGenerator:
    """
    Generates workflow YAML from natural language requirements.
    """
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary, agent: Optional[IAgent] = None):
        self.llm = llm_service
        self.library = smart_library
        self.agent = agent  # Optional agent to use instead of direct LLM calls
        logger.info("Workflow Generator initialized")
    
    def set_agent(self, agent: IAgent) -> None:
        """Set the agent after initialization."""
        self.agent = agent
    
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
        
        # If we have an agent, use it for generation
        if self.agent:
            return await self._generate_with_agent(requirements, domain, output_path)
        
        # Otherwise use the direct LLM approach
        return await self._generate_with_llm(requirements, domain, output_path)
    
    async def _generate_with_agent(
        self,
        requirements: str,
        domain: str,
        output_path: Optional[str] = None
    ) -> str:
        """Generate workflow using the agent."""
        prompt = f"""
        Create a workflow YAML based on these requirements:
        
        REQUIREMENTS:
        {requirements}
        
        DOMAIN: {domain}
        
        The workflow should follow our standard YAML format with steps for DEFINE, CREATE, and EXECUTE.
        Search for components in the library that might be useful for this workflow.
        
        Return ONLY the YAML content without additional text.
        """
        
        response = await self.agent.run(prompt)
        workflow_yaml = self._extract_yaml(response.result.text)
        
        # Save if path provided
        if output_path:
            await self._save_yaml(workflow_yaml, output_path)
            
        return workflow_yaml
    
    async def _generate_with_llm(
        self,
        requirements: str,
        domain: str,
        output_path: Optional[str] = None
    ) -> str:
        """Generate workflow using direct LLM calls."""
        # Get firmware for this domain
        firmware_record = await self.library.get_firmware(domain)
        firmware_content = firmware_record["code_snippet"] if firmware_record else ""
        
        # Get available records for this domain
        domain_agents = await self.library.find_records_by_domain(domain, "AGENT")
        domain_tools = await self.library.find_records_by_domain(domain, "TOOL")
        
        # Format available components for the prompt
        available_components = ""
        if domain_agents:
            available_components += "Available Agents:\n"
            for agent in domain_agents:
                available_components += f"- {agent['name']}: {agent['description']}\n"
        
        if domain_tools:
            available_components += "\nAvailable Tools:\n"
            for tool in domain_tools:
                available_components += f"- {tool['name']}: {tool['description']}\n"
        
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
        workflow_yaml = self._extract_yaml(workflow_yaml)
        
        # Save if path provided
        if output_path:
            await self._save_yaml(workflow_yaml, output_path)
            
        return workflow_yaml
    
    def _extract_yaml(self, text: str) -> str:
        """Extract YAML content from text that may contain other elements."""
        text = text.strip()
        
        # Remove markdown code block markers if present
        if "```yaml" in text:
            # Extract content between ```yaml and ```
            yaml_match = re.search(r"```yaml\s*([\s\S]*?)```", text)
            if yaml_match:
                text = yaml_match.group(1).strip()
        elif "```" in text:
            # Extract content between ``` and ```
            yaml_match = re.search(r"```\s*([\s\S]*?)```", text)
            if yaml_match:
                text = yaml_match.group(1).strip()
                
        return text
    
    async def _save_yaml(self, yaml_content: str, output_path: str) -> None:
        """Save YAML content to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            logger.info(f"Saved workflow to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving workflow: {str(e)}")