from typing import List, Dict, Any, Optional, Tuple
import yaml
import logging

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import RecordType

logger = logging.getLogger(__name__)

class WorkflowGenerator:
    """
    Generates workflow YAML from natural language requirements by consulting the Smart Library.
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
            requirements: Natural language description of requirements
            domain: Target domain for the workflow
            output_path: Optional path to save the YAML
            
        Returns:
            Generated YAML as string
        """
        # 1. First, analyze the requirements to identify needed capabilities
        capabilities = await self._extract_capabilities(requirements)
        
        # 2. For each capability, find suitable agents/tools from the library
        matching_items = await self._find_matching_items(capabilities, domain)
        
        # 3. Generate workflow YAML
        workflow_yaml = await self._generate_yaml(
            requirements=requirements,
            domain=domain,
            capabilities=capabilities,
            matching_items=matching_items
        )
        
        # 4. Optionally save the workflow
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(workflow_yaml)
            logger.info(f"Saved workflow to {output_path}")
            
        return workflow_yaml
    
    async def _extract_capabilities(self, requirements: str) -> List[str]:
        """
        Extract required capabilities from natural language requirements.
        
        Args:
            requirements: Natural language requirements
            
        Returns:
            List of capability descriptions
        """
        prompt = f"""
        Analyze the following requirements and extract a list of distinct capabilities needed:
        
        REQUIREMENTS:
        {requirements}
        
        Extract 3-7 key capabilities needed to fulfill these requirements. 
        Each capability should be a brief phrase (5-10 words).
        
        FORMAT YOUR RESPONSE AS A YAML LIST ONLY:
        - capability 1
        - capability 2
        etc.
        """
        
        response = await self.llm.generate(prompt)
        
        # Extract capabilities from YAML response
        try:
            capabilities = yaml.safe_load(response)
            if not isinstance(capabilities, list):
                logger.warning("LLM did not return a proper list, falling back to simple parsing")
                capabilities = [line.strip("- ").strip() for line in response.split("\n") if line.strip().startswith("-")]
                
            return capabilities
        except Exception as e:
            logger.warning(f"Error parsing capabilities: {e}")
            # Fallback: simple parsing
            return [line.strip("- ").strip() for line in response.split("\n") if line.strip().startswith("-")]
    
    async def _find_matching_items(
        self, 
        capabilities: List[str], 
        domain: str
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Find matching agents and tools for each capability.
        
        Args:
            capabilities: List of capability descriptions
            domain: Target domain
            
        Returns:
            Dictionary mapping capabilities to lists of (name, type, similarity) tuples
        """
        matches = {}
        
        for capability in capabilities:
            # Search for matching agents
            agent_matches = await self.library.semantic_search(
                query=capability,
                record_type=RecordType.AGENT,
                domain=domain,
                limit=3,
                threshold=0.3
            )
            
            # Search for matching tools
            tool_matches = await self.library.semantic_search(
                query=capability,
                record_type=RecordType.TOOL,
                domain=domain,
                limit=3,
                threshold=0.3
            )
            
            # Combine results
            matches[capability] = [
                (record.name, record.record_type.value, similarity)
                for record, similarity in agent_matches + tool_matches
            ]
            
        return matches
    
    async def _generate_yaml(
        self,
        requirements: str,
        domain: str,
        capabilities: List[str],
        matching_items: Dict[str, List[Tuple[str, str, float]]]
    ) -> str:
        """
        Generate workflow YAML based on requirements and matching items.
        
        Args:
            requirements: Original requirements
            domain: Target domain
            capabilities: List of capabilities
            matching_items: Dictionary of matching items for each capability
            
        Returns:
            Generated workflow YAML
        """
        # Get domain firmware and disclaimers
        firmware = await self.library.get_firmware(domain)
        
        # Format matches for the prompt
        matches_text = ""
        for capability, items in matching_items.items():
            matches_text += f"\nCapability: {capability}\n"
            for name, item_type, similarity in items:
                matches_text += f"- {item_type}: {name} (similarity: {similarity:.2f})\n"
        
        # Get domain-specific info for the prompt
        domains = await self.library.list_domains()
        domain_info = f"The target domain is: {domain}\n"
        domain_info += f"Available domains in the library: {', '.join(domains)}\n"
        
        # Format firmware information if available
        firmware_info = ""
        if firmware:
            firmware_info = f"""
            Domain firmware is available with the following rules:
            {firmware.code_snippet[:500]}... (truncated)
            """
        
        # Create the prompt
        prompt = f"""
        Generate a workflow YAML file based on the following requirements and available components.
        
        REQUIREMENTS:
        {requirements}
        
        DOMAIN INFORMATION:
        {domain_info}
        {firmware_info}
        
        MATCHING COMPONENTS FROM LIBRARY:
        {matches_text}
        
        The workflow should define the steps needed to accomplish the requirements using the available components. 
        If no perfect match exists, specify how to evolve an existing component or create a new one.
        
        Follow this YAML structure:
        ```yaml
        scenario_name: "A descriptive name"
        domain: "{domain}"
        description: "A brief description of what this workflow does"
        
        # Optional additional disclaimers
        additional_disclaimers:
          - "Any specific disclaimers for this scenario"
        
        steps:
          - type: "DEFINE"
            item_type: "[AGENT or TOOL]"
            name: "Name of the item"
            from_existing_snippet: "Name of existing item to reuse/evolve, if applicable"
            reuse_as_is: true/false
            evolve_changes:
              docstring_update: "Description of changes needed"
            description: "Description of the item's purpose"
            
          # Additional steps like CREATE and EXECUTE
          - type: "CREATE"
            item_type: "[AGENT or TOOL]"
            name: "Name of the item"
            
          - type: "EXECUTE"
            item_type: "[AGENT or TOOL]"
            name: "Name of the item"
            user_input: "Input to provide to the item"
        ```
        
        GENERATE ONLY VALID YAML WITH NO SURROUNDING TEXT OR CODE BLOCKS:
        """
        
        # Generate the workflow YAML
        yaml_content = await self.llm.generate(prompt)
        
        # Remove any markdown code block markers
        yaml_content = yaml_content.replace("```yaml", "").replace("```", "").strip()
        
        # Validate the YAML
        try:
            yaml.safe_load(yaml_content)
            logger.info("Generated valid YAML workflow")
        except Exception as e:
            logger.warning(f"Generated invalid YAML: {e}")
            
        return yaml_content