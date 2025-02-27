import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import yaml
import os
import asyncio

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import LibraryRecord, RecordType, RecordStatus
from evolving_agents.core.llm_service import LLMService
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.workflow.workflow_executor import WorkflowExecutor
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

class SystemAgent:
    """
    Central orchestrator for the evolving agents framework.
    Coordinates workflow execution, library management, and LLM interactions.
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
        
        # Initialize factories if not provided
        self.agent_factory = agent_factory or AgentFactory(smart_library, llm_service)
        self.tool_factory = tool_factory or ToolFactory(smart_library, llm_service)
        
        # Initialize workflow components
        self.workflow_generator = WorkflowGenerator(llm_service, smart_library)
        self.workflow_executor = WorkflowExecutor(
            smart_library, 
            llm_service, 
            self.agent_factory, 
            self.tool_factory
        )
        
        logger.info("SystemAgent initialized successfully")
        
    async def process_request(
        self, 
        request: str, 
        domain: str = "general", 
        output_yaml_path: Optional[str] = None,
        execute_workflow: bool = True
    ) -> Dict[str, Any]:
        """
        Process a natural language request by generating and optionally executing a workflow.
        
        Args:
            request: Natural language request
            domain: Domain for the request
            output_yaml_path: Optional path to save the generated workflow YAML
            execute_workflow: Whether to execute the generated workflow
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Processing request in domain '{domain}': {request[:100]}...")
        
        # 1. Generate workflow from request
        workflow_yaml = await self.workflow_generator.generate_workflow(
            requirements=request,
            domain=domain,
            output_path=output_yaml_path
        )
        
        result = {
            "workflow_yaml": workflow_yaml,
            "domain": domain,
            "request": request
        }
        
        # 2. Execute workflow if requested
        if execute_workflow:
            execution_results = await self.workflow_executor.execute_workflow(workflow_yaml)
            result["execution"] = execution_results
            
        return result
    
    async def process_yaml_workflow(self, yaml_path: str) -> Dict[str, Any]:
        """
        Process an existing YAML workflow file.
        
        Args:
            yaml_path: Path to YAML workflow file
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Processing workflow from file: {yaml_path}")
        
        # Read YAML file
        if not os.path.exists(yaml_path):
            error_msg = f"Workflow file not found: {yaml_path}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                workflow_yaml = f.read()
        except Exception as e:
            error_msg = f"Error reading workflow file: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
        # Execute workflow
        execution_results = await self.workflow_executor.execute_workflow(workflow_yaml)
        
        return {
            "workflow_yaml": workflow_yaml,
            "execution": execution_results
        }
    
    async def load_firmware(
        self, 
        firmware_content: str, 
        domain: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0.0"
    ) -> LibraryRecord:
        """
        Load firmware into the Smart Library.
        
        Args:
            firmware_content: The firmware rules content
            domain: Domain for the firmware
            name: Optional name (defaults to f"{domain}_firmware")
            description: Optional description
            version: Version string
            
        Returns:
            The firmware record
        """
        name = name or f"{domain}_firmware"
        description = description or f"Firmware rules for {domain} domain"
        
        # Generate embedding for the firmware
        embedding = await self.llm.embed(firmware_content)
        
        # Create firmware record
        firmware_record = LibraryRecord(
            name=name,
            record_type=RecordType.FIRMWARE,
            domain=domain,
            description=description,
            code_snippet=firmware_content,
            version=version,
            embedding=embedding,
            status=RecordStatus.ACTIVE,
            metadata={"type": "firmware", "domain": domain}
        )
        
        # Save to library
        await self.library.save_record(firmware_record)
        logger.info(f"Loaded firmware for domain '{domain}': {name}")
        
        return firmware_record
    
    async def load_initial_library(self, initial_records_path: str) -> List[LibraryRecord]:
        """
        Load initial records into the Smart Library from a YAML file.
        
        Args:
            initial_records_path: Path to YAML file with initial records
            
        Returns:
            List of loaded records
        """
        logger.info(f"Loading initial library from: {initial_records_path}")
        
        # Read YAML file
        try:
            with open(initial_records_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading initial records: {str(e)}")
            return []
            
        loaded_records = []
        
        # Process each record
        for record_data in data.get("records", []):
            try:
                record_type = RecordType(record_data.get("type", "TOOL").upper())
                domain = record_data.get("domain", "general")
                name = record_data.get("name")
                description = record_data.get("description", "")
                code_snippet = record_data.get("code", "")
                version = record_data.get("version", "1.0.0")
                tags = record_data.get("tags", [])
                
                # Skip if missing required fields
                if not name or not code_snippet:
                    logger.warning(f"Skipping record with missing required fields: {name}")
                    continue
                    
                # Generate embedding
                embedding = await self.llm.embed(code_snippet + " " + description)
                
                # Create record
                record = LibraryRecord(
                    name=name,
                    record_type=record_type,
                    domain=domain,
                    description=description,
                    code_snippet=code_snippet,
                    version=version,
                    embedding=embedding,
                    status=RecordStatus.ACTIVE,
                    metadata=record_data.get("metadata", {}),
                    tags=tags
                )
                
                # Save to library
                await self.library.save_record(record)
                loaded_records.append(record)
                
                logger.info(f"Loaded {record_type.value} '{name}' for domain '{domain}'")
                
            except Exception as e:
                logger.error(f"Error loading record: {str(e)}")
                
        logger.info(f"Loaded {len(loaded_records)} initial records")
        return loaded_records
    
    async def generate_agent_from_requirements(
        self,
        requirements: str,
        domain: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Tuple[LibraryRecord, Dict[str, Any]]:
        """
        Generate an agent directly from requirements without going through a workflow.
        
        Args:
            requirements: Natural language requirements
            domain: Domain for the agent
            name: Optional name for the agent
            description: Optional description
            
        Returns:
            Tuple of (agent record, generation metadata)
        """
        logger.info(f"Generating agent from requirements in domain '{domain}'")
        
        # Get firmware for domain
        firmware_record = await self.library.get_firmware(domain)
        firmware_content = firmware_record.code_snippet if firmware_record else ""
        
        # Generate a name if not provided
        if not name:
            name_prompt = f"Generate a concise, descriptive name for an AI agent with these requirements: {requirements}"
            name = await self.llm.generate(name_prompt)
            name = name.strip().replace(" ", "_")
            
        # Generate a description if not provided
        if not description:
            desc_prompt = f"Write a one-sentence description for an AI agent that: {requirements}"
            description = await self.llm.generate(desc_prompt)
            
        # Generate agent code
        generation_prompt = f"""
        {firmware_content}
        
        REQUIREMENTS:
        {requirements}
        
        Create a complete, self-contained agent that fulfills these requirements.
        The agent should be named '{name}' and handle all aspects of the requirements.
        
        Ensure the agent follows all firmware rules for the {domain} domain.
        Include appropriate docstrings, error handling, and domain-specific disclaimers.
        
        RETURN ONLY THE AGENT CODE:
        """
        
        agent_code = await self.llm.generate(generation_prompt)
        
        # Generate embedding
        embedding = await self.llm.embed(agent_code + " " + description + " " + requirements)
        
        # Create agent record
        agent_record = LibraryRecord(
            name=name,
            record_type=RecordType.AGENT,
            domain=domain,
            description=description,
            code_snippet=agent_code,
            version="1.0.0",
            embedding=embedding,
            status=RecordStatus.ACTIVE,
            metadata={
                "generated_from": "requirements",
                "original_requirements": requirements
            },
            tags=[domain, "generated", "requirements-based"]
        )
        
        # Save to library
        await self.library.save_record(agent_record)
        logger.info(f"Generated agent '{name}' from requirements")
        
        # Metadata about the generation process
        metadata = {
            "agent_id": agent_record.id,
            "name": name,
            "domain": domain,
            "description": description,
            "requirements": requirements
        }
        
        return agent_record, metadata
    
    async def semantic_find_and_execute(
        self,
        query: str,
        input_text: str,
        domain: Optional[str] = None,
        record_type: RecordType = RecordType.AGENT,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Find the most semantically similar agent or tool and execute it.
        If no suitable match is found, generate a new one.
        
        Args:
            query: Query to find matching agent/tool
            input_text: Input to provide to the agent/tool
            domain: Optional domain filter
            record_type: Type of record to find (AGENT or TOOL)
            similarity_threshold: Threshold for considering a match suitable
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Semantic find and execute for query: {query}")
        
        # Search for matching records
        search_results = await self.library.semantic_search(
            query=query,
            record_type=record_type,
            domain=domain,
            limit=1,
            threshold=0.0  # Get top match regardless of score
        )
        
        if search_results and search_results[0][1] >= similarity_threshold:
            # Use existing record
            record, similarity = search_results[0]
            logger.info(f"Found matching {record_type.value} '{record.name}' (similarity: {similarity:.3f})")
            
            # Create instance
            if record_type == RecordType.AGENT:
                instance = await self.agent_factory.create_agent(record)
                result = await self.agent_factory.execute_agent(instance, input_text)
            else:  # TOOL
                instance = await self.tool_factory.create_tool(record)
                result = await self.tool_factory.execute_tool(instance, input_text)
                
            # Update usage metrics
            await self.library.update_usage_metrics(record.id, success=True)
            
            return {
                "status": "success",
                "action": "execute_existing",
                "record_id": record.id,
                "name": record.name,
                "similarity": similarity,
                "result": result
            }
        else:
            # No suitable match found, generate new
            if search_results:
                best_match, similarity = search_results[0]
                logger.info(f"Best match '{best_match.name}' below threshold (similarity: {similarity:.3f})")
            else:
                logger.info("No matches found")
                
            # Generate new agent/tool from requirements
            if record_type == RecordType.AGENT:
                record, metadata = await self.generate_agent_from_requirements(
                    requirements=query,
                    domain=domain or "general"
                )
                
                # Execute the new agent
                instance = await self.agent_factory.create_agent(record)
                result = await self.agent_factory.execute_agent(instance, input_text)
            else:
                # Similar implementation for tools
                raise NotImplementedError("Tool generation not implemented yet")
                
            # Update usage metrics
            await self.library.update_usage_metrics(record.id, success=True)
            
            return {
                "status": "success",
                "action": "generate_and_execute",
                "record_id": record.id,
                "name": record.name,
                "result": result
            }
    
    async def initialize_system(self, config_path: str) -> Dict[str, Any]:
        """
        Initialize the system with configuration from a YAML file.
        Sets up firmware, loads initial records, and configures system parameters.
        
        Args:
            config_path: Path to system configuration YAML
            
        Returns:
            Dictionary with initialization results
        """
        logger.info(f"Initializing system from config: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading system configuration: {str(e)}")
            return {"status": "error", "message": f"Configuration error: {str(e)}"}
            
        results = {
            "firmware_loaded": [],
            "records_loaded": 0,
            "domains_configured": []
        }
        
        # Load firmware for each domain
        for firmware_config in config.get("firmware", []):
            domain = firmware_config.get("domain", "general")
            
            # Load firmware content from file or inline
            if "file" in firmware_config:
                try:
                    with open(firmware_config["file"], 'r', encoding='utf-8') as f:
                        firmware_content = f.read()
                except Exception as e:
                    logger.error(f"Error loading firmware file: {str(e)}")
                    continue
            else:
                firmware_content = firmware_config.get("content", "")
                
            if not firmware_content:
                logger.warning(f"Empty firmware content for domain '{domain}'")
                continue
                
            # Load firmware
            firmware_record = await self.load_firmware(
                firmware_content=firmware_content,
                domain=domain,
                name=firmware_config.get("name"),
                description=firmware_config.get("description"),
                version=firmware_config.get("version", "1.0.0")
            )
            
            results["firmware_loaded"].append({
                "domain": domain,
                "name": firmware_record.name,
                "id": firmware_record.id
            })
            results["domains_configured"].append(domain)
            
        # Load initial records
        for records_file in config.get("initial_records", []):
            loaded = await self.load_initial_library(records_file)
            results["records_loaded"] += len(loaded)
            
        logger.info(f"System initialization complete: {len(results['firmware_loaded'])} firmware records, "
                   f"{results['records_loaded']} initial records")
        
        return {
            "status": "success",
            "initialization": results
        }