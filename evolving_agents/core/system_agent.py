# evolving_agents/core/system_agent.py

import logging
import os
import yaml
from typing import Dict, List, Any, Optional, Union, Tuple

from beeai_framework.backend.message import UserMessage

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.firmware.firmware_manager import FirmwareManager
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.workflow.workflow_executor import WorkflowExecutor
from evolving_agents.smart_library.record import RecordType

logger = logging.getLogger(__name__)

class SystemAgent:
    """
    Central orchestrator for the evolving agents framework.
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
        self.firmware_manager = FirmwareManager(smart_library)
        
        # Initialize factories
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
        
        logger.info("SystemAgent initialized")
    
    async def initialize_system(self, config_path: str) -> Dict[str, Any]:
        """
        Initialize the system from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration
            
        Returns:
            Initialization results
        """
        logger.info(f"Initializing system from: {config_path}")
        
        if not os.path.exists(config_path):
            error = f"Configuration file not found: {config_path}"
            logger.error(error)
            return {"status": "error", "message": error}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            error = f"Error loading configuration: {str(e)}"
            logger.error(error)
            return {"status": "error", "message": error}
        
        results = {
            "firmware_loaded": [],
            "initial_records": 0,
            "domains": []
        }
        
        # Load firmware definitions
        for firmware_file in config.get("firmware_files", []):
            if os.path.exists(firmware_file):
                firmware_ids = await self.firmware_manager.load_firmware_from_yaml(firmware_file)
                for domain, fw_id in firmware_ids.items():
                    results["firmware_loaded"].append({"domain": domain, "id": fw_id})
                    if domain not in results["domains"]:
                        results["domains"].append(domain)
        
        # Load initial records
        for records_file in config.get("initial_records", []):
            if os.path.exists(records_file):
                try:
                    with open(records_file, 'r', encoding='utf-8') as f:
                        records_data = yaml.safe_load(f)
                    
                    for record_data in records_data.get("records", []):
                        # Implement record loading logic here
                        pass
                    
                    results["initial_records"] += len(records_data.get("records", []))
                except Exception as e:
                    logger.error(f"Error loading records from {records_file}: {str(e)}")
        
        return {"status": "success", "results": results}
    
    async def process_request(
        self, 
        request: str, 
        domain: str = "general", 
        output_yaml_path: Optional[str] = None,
        execute_workflow: bool = True
    ) -> Dict[str, Any]:
        """
        Process a natural language request by generating and executing a workflow.
        
        Args:
            request: Natural language request
            domain: Domain for the request
            output_yaml_path: Optional path to save the generated YAML
            execute_workflow: Whether to execute the workflow
            
        Returns:
            Results dictionary
        """
        logger.info(f"Processing request in domain '{domain}': {request[:100]}...")
        
        # Generate workflow YAML
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
        
        # Execute workflow if requested
        if execute_workflow:
            execution_results = await self.workflow_executor.execute_workflow(workflow_yaml)
            result["execution"] = execution_results
        
        return result
    
    async def process_yaml_workflow(self, yaml_path: str) -> Dict[str, Any]:
        """
        Process a YAML workflow file.
        
        Args:
            yaml_path: Path to YAML workflow
            
        Returns:
            Execution results
        """
        logger.info(f"Processing workflow from: {yaml_path}")
        
        if not os.path.exists(yaml_path):
            error = f"Workflow file not found: {yaml_path}"
            logger.error(error)
            return {"status": "error", "message": error}
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                workflow_yaml = f.read()
        except Exception as e:
            error = f"Error reading workflow file: {str(e)}"
            logger.error(error)
            return {"status": "error", "message": error}
        
        execution_results = await self.workflow_executor.execute_workflow(workflow_yaml)
        
        return {
            "status": "success",
            "workflow_yaml": workflow_yaml,
            "execution": execution_results
        }