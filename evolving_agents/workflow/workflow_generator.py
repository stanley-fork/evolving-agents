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

    def set_llm_service(self, llm_service):
        """Set the LLM service."""
        self.llm = llm_service
    
    def set_agent(self, agent):
        """Set the agent for this workflow generator."""
        self.agent = agent


    async def generate_workflow_from_design_with_capabilities(
        self,
        workflow_design: Dict[str, Any],
        library_entries: Dict[str, Any],
        capability_mapping: Dict[str, Dict[str, Any]] = None,
        domain: str = "general"
    ) -> str:
        """
        Generate a workflow YAML from the design with capability-based component selection.
        
        Args:
            workflow_design: The workflow design specification
            library_entries: Library entries for reuse, evolution, or creation
            capability_mapping: Mapping from capability IDs to components
            domain: Domain for the workflow
            
        Returns:
            YAML workflow as a string
        """
        # Fix: Use self.llm instead of self.llm_service
        if not self.llm:
            # Fall back to standard workflow generation if no LLM available
            return await self.generate_workflow_from_design(workflow_design, library_entries)
        
        try:
            # Format the workflow design for the prompt
            workflow_json = json.dumps(workflow_design, indent=2)
            
            # Format library entries
            library_json = json.dumps(library_entries, indent=2)
            
            # Format capability mapping
            capability_info = ""
            if capability_mapping:
                capability_details = []
                for cap_id, component in capability_mapping.items():
                    cap_info = f"Capability '{cap_id}' can be provided by component '{component['name']}'"
                    
                    # Add details about the component
                    cap_info += f" (ID: {component['id']}, Type: {component['record_type']})"
                    
                    # Add information about how the component provides this capability
                    for cap in component.get("capabilities", []):
                        if cap.get("id") == cap_id:
                            cap_info += f"\n - Description: {cap.get('description', 'No description')}"
                            if "context" in cap:
                                context = cap["context"]
                                required = context.get("required_fields", [])
                                produced = context.get("produced_fields", [])
                                if required:
                                    # Handle both list and string formats
                                    if isinstance(required, list):
                                        cap_info += f"\n - Requires: {', '.join(required)}"
                                    else:
                                        cap_info += f"\n - Requires: {required}"
                                if produced:
                                    # Handle both list and string formats
                                    if isinstance(produced, list):
                                        cap_info += f"\n - Produces: {', '.join(produced)}"
                                    else:
                                        cap_info += f"\n - Produces: {produced}"
                    
                    capability_details.append(cap_info)
                
                capability_info = "CAPABILITY MAPPING:\n" + "\n".join(capability_details)
            else:
                capability_info = "No capability mapping provided. You'll need to create new components."
            
            # Create the prompt
            prompt = f"""
            Generate a complete YAML workflow based on this design, library entries, and capability mapping.
            
            WORKFLOW DESIGN:
            {workflow_json}
            
            LIBRARY ENTRIES:
            {library_json}
            
            {capability_info}
            
            DOMAIN: {domain}
            
            Create a complete YAML workflow with these sections:
            1. scenario_name: A clear name for the workflow
            2. domain: The domain of the workflow ({domain})
            3. description: A detailed description of what the workflow does
            4. steps: The sequence of operations to perform
            
            Each step should have:
            - type: One of "DEFINE", "CREATE", or "EXECUTE"
            - item_type: Type of item (AGENT or TOOL)
            - name: Name of the item
            - Additional fields as needed
            
            For DEFINE steps:
            - Add code_snippet for new components
            - For components in the capability mapping, reference them by name
            
            For CREATE steps:
            - Reference the previously defined components
            
            For EXECUTE steps:
            - Specify the component to execute and the input data
            
            VERY IMPORTANT: When including sample data like invoice text, always wrap it as follows:
            
            user_input: |
            Sample text goes here
            with proper indentation
            for multi-line content
            
            The workflow should implement all the required functionality while leveraging the capability mapping to reuse existing components wherever possible.
            
            Return only the YAML content without additional comments or explanations.
            """
            
            # Generate the workflow
            response = await self.llm.generate(prompt)  # Fix: using self.llm instead of llm_service
            
            # Extract the YAML content
            yaml_content = response
            
            # Clean up if needed (extract from markdown code blocks, etc.)
            if "```yaml" in response:
                yaml_content = response.split("```yaml")[1].split("```")[0].strip()
            elif "```" in response:
                yaml_content = response.split("```")[1].split("```")[0].strip()
            
            return yaml_content
            
        except Exception as e:
            logger.error(f"Error generating workflow with capabilities: {str(e)}")
            # Fall back to standard generation on error
            return await self.generate_workflow_from_design(workflow_design, library_entries)
    
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