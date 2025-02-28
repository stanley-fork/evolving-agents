# evolving_agents/workflow/workflow_executor.py

import logging
import yaml
from typing import Dict, List, Any, Optional

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory
from evolving_agents.smart_library.record import LibraryRecord, RecordType, RecordStatus

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """
    Executes YAML workflows by instantiating and running agents and tools.
    """
    def __init__(
        self, 
        smart_library: SmartLibrary, 
        llm_service: LLMService,
        agent_factory: AgentFactory,
        tool_factory: ToolFactory
    ):
        self.library = smart_library
        self.llm = llm_service
        self.agent_factory = agent_factory
        self.tool_factory = tool_factory
        
        # Runtime registry
        self.active_items = {}
    
    async def execute_workflow(self, workflow_yaml: str) -> Dict[str, Any]:
        """
        Execute a workflow from YAML.
        
        Args:
            workflow_yaml: YAML string defining the workflow
            
        Returns:
            Execution results
        """
        logger.info("Executing workflow")
        
        # Parse YAML
        try:
            workflow = yaml.safe_load(workflow_yaml)
        except Exception as e:
            error = f"Error parsing workflow YAML: {str(e)}"
            logger.error(error)
            return {"status": "error", "message": error}
        
        # Extract metadata
        scenario_name = workflow.get("scenario_name", "Unnamed Scenario")
        domain = workflow.get("domain", "general")
        disclaimers = workflow.get("additional_disclaimers", [])
        
        logger.info(f"Executing scenario: {scenario_name} in domain: {domain}")
        
        # Get firmware for this domain
        firmware_record = await self.library.get_firmware(domain)
        firmware_content = firmware_record.code_snippet if firmware_record else ""
        
        # Execute steps
        results = {"steps": [], "scenario_name": scenario_name, "domain": domain}
        
        for i, step in enumerate(workflow.get("steps", [])):
            step_type = step.get("type", "").upper()
            logger.info(f"Executing step {i+1}: {step_type}")
            
            step_result = {"type": step_type, "status": "pending"}
            
            try:
                if step_type == "DEFINE":
                    step_result = await self._execute_define_step(
                        step, domain, firmware_content, disclaimers
                    )
                elif step_type == "CREATE":
                    step_result = await self._execute_create_step(
                        step, domain, firmware_content
                    )
                elif step_type == "EXECUTE":
                    step_result = await self._execute_execute_step(step)
                else:
                    step_result = {"status": "error", "message": f"Unknown step type: {step_type}"}
            except Exception as e:
                logger.error(f"Error executing step {i+1}: {str(e)}")
                step_result = {"status": "error", "message": f"Error: {str(e)}"}
            
            results["steps"].append(step_result)
            
            # Stop if step failed
            if step_result.get("status") == "error":
                logger.warning(f"Workflow execution stopped due to error in step {i+1}")
                results["status"] = "error"
                results["message"] = f"Failed at step {i+1}: {step_result.get('message')}"
                break
        
        if "status" not in results:
            results["status"] = "success"
            results["message"] = f"Successfully executed workflow '{scenario_name}'"
        
        return results
    
    async def _execute_define_step(
        self,
        step: Dict[str, Any],
        domain: str,
        firmware_content: str,
        disclaimers: List[str]
    ) -> Dict[str, Any]:
        """Execute a DEFINE step to define an agent or tool."""
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        description = step.get("description", "")
        
        logger.info(f"Defining {item_type} '{name}'")
        
        # Check if we need to reuse or evolve an existing item
        if "from_existing_snippet" in step:
            source_name = step.get("from_existing_snippet")
            source_record = await self.library.find_record_by_name(source_name)
            
            if not source_record:
                return {"status": "error", "message": f"Source item not found: {source_name}"}
            
            if step.get("reuse_as_is", False):
                # Reuse the existing item as is
                logger.info(f"Reusing {source_name} as {name}")
                
                # Create a new record with the same code
                new_record = LibraryRecord(
                    name=name,
                    record_type=RecordType(item_type),
                    domain=domain,
                    description=description or source_record.description,
                    code_snippet=source_record.code_snippet,
                    version="1.0.0",
                    status=RecordStatus.ACTIVE,
                    metadata={
                        "reused_from": source_record.id,
                        "disclaimers": disclaimers
                    }
                )
                
                # Save to library
                record_id = await self.library.save_record(new_record)
                
                return {
                    "status": "success",
                    "action": "reuse",
                    "message": f"Reused {source_name} as {name}",
                    "record_id": record_id
                }
            else:
                # Evolve the existing item
                logger.info(f"Evolving {source_name} to {name}")
                
                evolve_changes = step.get("evolve_changes", {})
                docstring_update = evolve_changes.get("docstring_update", "")
                
                # Generate evolved code
                evolve_prompt = f"""
                {firmware_content}
                
                ORIGINAL CODE:
                ```
                {source_record.code_snippet}
                ```
                
                REQUIRED CHANGES:
                - New name: {name}
                - Description: {description or source_record.description}
                - Docstring update: {docstring_update}
                
                DOMAIN REQUIREMENTS:
                - Domain: {domain}
                
                REQUIRED DISCLAIMERS:
                {chr(10).join(disclaimers)}
                
                Evolve the code to implement these changes while maintaining core functionality.
                Include all required disclaimers and follow domain guidelines.
                
                EVOLVED CODE:
                """
                
                evolved_code = await self.llm.generate(evolve_prompt)
                
                # Create a new evolved record
                new_record = LibraryRecord(
                    name=name,
                    record_type=RecordType(item_type),
                    domain=domain,
                    description=description or source_record.description,
                    code_snippet=evolved_code,
                    version="1.0.0",
                    status=RecordStatus.ACTIVE,
                    metadata={
                        "evolved_from": source_record.id,
                        "evolution_changes": evolve_changes,
                        "disclaimers": disclaimers
                    }
                )
                
                # Save to library
                record_id = await self.library.save_record(new_record)
                
                return {
                    "status": "success",
                    "action": "evolve",
                    "message": f"Evolved {source_name} to {name}",
                    "record_id": record_id
                }
        else:
            # Create a new item from scratch
            logger.info(f"Creating new {item_type} {name}")
            
            # Generate code for the new item
            create_prompt = f"""
            {firmware_content}
            
            CREATE A NEW {item_type}:
            Name: {name}
            Description: {description}
            
            DOMAIN: {domain}
            
            REQUIRED DISCLAIMERS:
            {chr(10).join(disclaimers)}
            
            Generate complete code for this {item_type.lower()} that follows all guidelines.
            Include proper docstrings and all required disclaimers.
            
            CODE:
            """
            
            new_code = await self.llm.generate(create_prompt)
            
            # Create a new record
            new_record = LibraryRecord(
                name=name,
                record_type=RecordType(item_type),
                domain=domain,
                description=description,
                code_snippet=new_code,
                version="1.0.0",
                status=RecordStatus.ACTIVE,
                metadata={
                    "created_from": "workflow",
                    "disclaimers": disclaimers
                }
            )
            
            # Save to library
            record_id = await self.library.save_record(new_record)
            
            return {
                "status": "success",
                "action": "create",
                "message": f"Created new {item_type} {name}",
                "record_id": record_id
            }
    
    async def _execute_create_step(
        self,
        step: Dict[str, Any],
        domain: str,
        firmware_content: str
    ) -> Dict[str, Any]:
        """Execute a CREATE step to instantiate an agent or tool."""
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        
        logger.info(f"Creating {item_type} instance '{name}'")
        
        # Find the record
        record = await self.library.find_record_by_name(name)
        
        if not record:
            return {"status": "error", "message": f"Item not found: {name}"}
        
        if record.record_type.value != item_type:
            return {"status": "error", "message": f"Item {name} is not a {item_type}"}
        
        # Create instance
        try:
            if item_type == "AGENT":
                instance = await self.agent_factory.create_agent(
                    record=record,
                    firmware_content=firmware_content
                )
            else:  # TOOL
                instance = await self.tool_factory.create_tool(
                    record=record,
                    firmware_content=firmware_content
                )
                
            # Store in active items
            self.active_items[name] = {
                "record": record,
                "instance": instance,
                "type": item_type
            }
            
            return {
                "status": "success",
                "message": f"Created {item_type} instance {name}",
                "record_id": record.id
            }
        except Exception as e:
            logger.error(f"Error creating {item_type} {name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error creating {item_type} {name}: {str(e)}"
            }
    
    async def _execute_execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an EXECUTE step to run an agent or tool."""
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        user_input = step.get("user_input", "")
        
        logger.info(f"Executing {item_type} '{name}' with input: {user_input[:50]}...")
        
        # Check if item is active
        if name not in self.active_items:
            return {"status": "error", "message": f"{item_type} {name} is not active"}
        
        active_item = self.active_items[name]
        
        # Execute the item
        try:
            if item_type == "AGENT":
                result = await self.agent_factory.execute_agent(
                    agent_instance=active_item["instance"],
                    input_text=user_input
                )
            else:  # TOOL
                result = await self.tool_factory.execute_tool(
                    tool_instance=active_item["instance"],
                    input_text=user_input
                )
                
            # Update usage metrics
            await self.library.update_usage_metrics(active_item["record"].id, True)
            
            return {
                "status": "success",
                "message": f"Executed {item_type} {name}",
                "result": result,
                "record_id": active_item["record"].id
            }
        except Exception as e:
            logger.error(f"Error executing {item_type} {name}: {str(e)}")
            
            # Update usage metrics as failure
            await self.library.update_usage_metrics(active_item["record"].id, False)
            
            return {
                "status": "error",
                "message": f"Error executing {item_type} {name}: {str(e)}"
            }