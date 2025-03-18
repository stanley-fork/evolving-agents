# evolving_agents/workflow/workflow_generator.py

import json
import logging
from typing import Dict, Any, List, Optional

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary

logger = logging.getLogger(__name__)

class WorkflowGenerator:
    """
    Generator for creating workflows from requirements.
    """
    def __init__(self, llm_service: LLMService, smart_library: SmartLibrary):
        """
        Initialize the workflow generator.
        
        Args:
            llm_service: LLM service for text generation
            smart_library: Smart library for component lookup
        """
        self.llm = llm_service
        self.library = smart_library
        self.agent = None
        logger.info("Workflow Generator initialized")
    
    def set_agent(self, agent):
        """Set the agent for this workflow generator."""
        self.agent = agent
    
    async def generate_workflow_from_design(
        self,
        workflow_design: Dict[str, Any],
        library_entries: Dict[str, Any]
    ) -> str:
        """
        Generate a complete YAML workflow from a workflow design and library entries.
        
        Args:
            workflow_design: The workflow design dictionary
            library_entries: The library entries dictionary
            
        Returns:
            Complete YAML workflow as a string
        """
        task_objective = workflow_design.get("task_objective", "Task")
        domain = workflow_design.get("domain", "general")
        
        # Combine all components from library entries
        reuse_components = library_entries.get("reuse", [])
        evolve_components = library_entries.get("evolve", [])
        create_components = library_entries.get("create", [])
        
        # Generate a workflow generation prompt
        workflow_prompt = f"""
        Generate a complete YAML workflow for the following task:
        
        TASK OBJECTIVE:
        {task_objective}
        
        DOMAIN:
        {domain}
        
        COMPONENTS TO REUSE:
        {json.dumps(reuse_components, indent=2)}
        
        COMPONENTS TO EVOLVE:
        {json.dumps(evolve_components, indent=2)}
        
        COMPONENTS TO CREATE:
        {json.dumps(create_components, indent=2)}
        
        WORKFLOW LOGIC:
        {json.dumps(workflow_design.get("sequence", []), indent=2)}
        
        DATA FLOW:
        {json.dumps(workflow_design.get("data_flow", []), indent=2)}
        
        The YAML workflow should follow this structure:
        ```yaml
        scenario_name: [task name]
        domain: [domain]
        description: [description]
        
        steps:
          # First define components that need to be created from scratch
          - type: "DEFINE"
            item_type: "AGENT" or "TOOL" 
            name: [component name]
            description: [component description]
            code_snippet: [component code]
          
          # Then define components that need to be evolved
          - type: "DEFINE"
            item_type: "AGENT" or "TOOL"
            name: [component name]
            from_existing: [existing component name or ID]
            description: [component description]
            code_snippet: [component code]
          
          # Then create all components
          - type: "CREATE"
            item_type: "AGENT" or "TOOL"
            name: [component name]
          
          # Finally execute the workflow
          - type: "EXECUTE"
            item_type: "AGENT"
            name: [component name]
            user_input: [input text]
        ```
        
        Return only the YAML content without any explanation.
        """
        
        # Generate the workflow
        yaml_workflow = await self.llm.generate(workflow_prompt)
        
        # Clean up the response to extract just the YAML
        if "```yaml" in yaml_workflow and "```" in yaml_workflow:
            yaml_content = yaml_workflow.split("```yaml")[1].split("```")[0].strip()
            return yaml_content
        
        # If the response doesn't contain YAML code blocks, return as is
        return yaml_workflow