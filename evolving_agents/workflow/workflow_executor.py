import yaml
import logging
from typing import Dict, List, Any, Optional

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.smart_library.record import RecordType, RecordStatus, LibraryRecord
from evolving_agents.core.llm_service import LLMService
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory
from evolving_agents.smart_library.record_validator import validate_record

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """
    Executes workflow YAML by retrieving, evolving, or creating agents and tools.
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
        
        # Runtime registry for the current workflow
        self.active_items = {}
        
    async def execute_workflow(self, workflow_yaml: str) -> Dict[str, Any]:
        """
        Execute a workflow from YAML.
        
        Args:
            workflow_yaml: YAML string defining the workflow
            
        Returns:
            Dictionary with workflow results
        """
        # Parse YAML
        try:
            workflow = yaml.safe_load(workflow_yaml)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing workflow YAML: {e}")
            return {"status": "error", "message": f"Invalid YAML: {str(e)}"}
        
        # Extract workflow metadata
        scenario_name = workflow.get("scenario_name", "Unnamed Scenario")
        domain = workflow.get("domain", "general")
        description = workflow.get("description", "")
        disclaimers = workflow.get("additional_disclaimers", [])
        
        logger.info(f"Executing workflow: {scenario_name} ({domain})")
        
        # Get firmware for the domain
        firmware_record = await self.library.get_firmware(domain)
        if not firmware_record:
            logger.warning(f"No firmware found for domain {domain}, using empty firmware")
            firmware_content = ""
        else:
            firmware_content = firmware_record.code_snippet
        
        # Execute steps
        results = {"steps": [], "scenario_name": scenario_name, "domain": domain}
        
        for i, step in enumerate(workflow.get("steps", [])):
            step_type = step.get("type", "").upper()
            logger.info(f"Executing step {i+1}: {step_type}")
            
            if step_type == "DEFINE":
                step_result = await self._execute_define_step(step, domain, firmware_content, disclaimers)
            elif step_type == "CREATE":
                step_result = await self._execute_create_step(step, domain, firmware_content)
            elif step_type == "EXECUTE":
                step_result = await self._execute_execute_step(step)
            else:
                step_result = {"status": "error", "message": f"Unknown step type: {step_type}"}
                
            results["steps"].append(step_result)
            
            # Stop if a step fails
            if step_result.get("status") == "error":
                logger.warning(f"Workflow execution stopped due to error in step {i+1}")
                break
                
        logger.info(f"Workflow execution completed: {scenario_name}")
        return results
    
    async def _execute_define_step(
        self, 
        step: Dict[str, Any], 
        domain: str, 
        firmware_content: str,
        disclaimers: List[str]
    ) -> Dict[str, Any]:
        """
        Execute a DEFINE step, which defines an agent or tool by reusing or evolving.
        
        Args:
            step: Step configuration
            domain: Domain for the step
            firmware_content: Firmware content for the domain
            disclaimers: Additional disclaimers
            
        Returns:
            Dictionary with step results
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        description = step.get("description", "")
        
        # Check if this is reusing or evolving an existing item
        if "from_existing_snippet" in step:
            source_name = step.get("from_existing_snippet")
            source_record = await self.library.find_record_by_name(source_name)
            
            if not source_record:
                return {"status": "error", "message": f"Source item not found: {source_name}"}
                
            reuse_as_is = step.get("reuse_as_is", False)
            
            if reuse_as_is:
                # Reuse the existing item as is
                logger.info(f"Reusing {source_name} as {name}")
                
                # Create a new record with the same code but different metadata
                new_record = LibraryRecord(
                    name=name,
                    record_type=RecordType(item_type),
                    domain=domain,
                    description=description or source_record.description,
                    code_snippet=source_record.code_snippet,
                    version="1.0.0",
                    embedding=source_record.embedding.copy() if source_record.embedding else None,
                    status=RecordStatus.ACTIVE,
                    metadata={
                        "reused_from": source_record.id,
                        "original_name": source_record.name,
                        "disclaimers": disclaimers
                    },
                    tags=source_record.tags.copy() if source_record.tags else []
                )
                
                # Save to library
                await self.library.save_record(new_record)
                
                return {
                    "status": "success",
                    "action": "reuse",
                    "record_id": new_record.id,
                    "name": name,
                    "message": f"Reused {source_name} as {name}"
                }
            else:
                # Evolve the existing item
                logger.info(f"Evolving {source_name} to {name}")
                
                evolve_changes = step.get("evolve_changes", {})
                docstring_update = evolve_changes.get("docstring_update", "")
                
                # Create evolution prompt
                evolution_prompt = f"""
                {firmware_content}
                
                ORIGINAL CODE:
                ```
                {source_record.code_snippet}
                ```
                
                REQUESTED CHANGES:
                - New name: {name}
                - New description: {description or source_record.description}
                - Docstring update: {docstring_update}
                
                REQUIRED DISCLAIMERS:
                {chr(10).join(disclaimers)}
                
                Please evolve the code to implement these changes while maintaining its core functionality.
                Ensure all firmware rules are followed and disclaimers are included.
                
                RETURN ONLY THE EVOLVED CODE:
                """
                
                # Generate evolved code
                evolved_code = await self.llm.generate(evolution_prompt)
                
                # Validate the evolved code
                validation_result = validate_record(
                    code=evolved_code,
                    record_type=RecordType(item_type),
                    domain=domain,
                    firmware_content=firmware_content,
                    required_disclaimers=disclaimers
                )
                
                if not validation_result["valid"]:
                    # Try one more time with the validation feedback
                    issues = chr(10).join(validation_result["issues"])
                    retry_prompt = f"""
                    {firmware_content}
                    
                    Your previous code had the following issues:
                    {issues}
                    
                    Please fix these issues in the code:
                    ```
                    {evolved_code}
                    ```
                    
                    Ensure all firmware rules are followed and required disclaimers are included.
                    
                    RETURN ONLY THE FIXED CODE:
                    """
                    
                    evolved_code = await self.llm.generate(retry_prompt)
                    validation_result = validate_record(
                        code=evolved_code,
                        record_type=RecordType(item_type),
                        domain=domain,
                        firmware_content=firmware_content,
                        required_disclaimers=disclaimers
                    )
                
                # Create a new record with the evolved code
                new_record = await self.library.evolve_record(
                    parent_id=source_record.id,
                    new_code_snippet=evolved_code,
                    description=description or source_record.description,
                    status=RecordStatus.ACTIVE if validation_result["valid"] else RecordStatus.PENDING
                )
                
                # Add evolution metadata
                new_record.name = name  # Override the name
                new_record.metadata["evolution_type"] = "workflow_evolution"
                new_record.metadata["evolution_changes"] = evolve_changes
                new_record.metadata["validation_result"] = validation_result
                
                # Re-save with updated metadata
                await self.library.save_record(new_record)
                
                return {
                    "status": "success",
                    "action": "evolve",
                    "record_id": new_record.id,
                    "name": name,
                    "validation": validation_result,
                    "message": f"Evolved {source_name} to {name}"
                }
        else:
            # Create a new item from scratch
            logger.info(f"Defining new {item_type} {name} from scratch")
            
            creation_prompt = f"""
            {firmware_content}
            
            CREATE A NEW {item_type}:
            Name: {name}
            Description: {description}
            
            REQUIRED DISCLAIMERS:
            {chr(10).join(disclaimers)}
            
            Please generate code for a new {item_type.lower()} that fulfills the description.
            Ensure all firmware rules are followed and disclaimers are included.
            
            RETURN ONLY THE CODE:
            """
            
            # Generate new code
            new_code = await self.llm.generate(creation_prompt)
            
            # Validate the new code
            validation_result = validate_record(
                code=new_code,
                record_type=RecordType(item_type),
                domain=domain,
                firmware_content=firmware_content,
                required_disclaimers=disclaimers
            )
            
            if not validation_result["valid"]:
                # Try one more time with the validation feedback
                issues = chr(10).join(validation_result["issues"])
                retry_prompt = f"""
                {firmware_content}
                
                Your previous code had the following issues:
                {issues}
                
                Please fix these issues in the code:
                ```
                {new_code}
                ```
                
                Ensure all firmware rules are followed and required disclaimers are included.
                
                RETURN ONLY THE FIXED CODE:
                """
                
                new_code = await self.llm.generate(retry_prompt)
                validation_result = validate_record(
                    code=new_code,
                    record_type=RecordType(item_type),
                    domain=domain,
                    firmware_content=firmware_content,
                    required_disclaimers=disclaimers
                )
            
            # Create embedding for the new code
            embedding = await self.llm.embed(new_code + " " + description)
            
            # Create a new record
            new_record = LibraryRecord(
                name=name,
                record_type=RecordType(item_type),
                domain=domain,
                description=description,
                code_snippet=new_code,
                version="1.0.0",
                embedding=embedding,
                status=RecordStatus.ACTIVE if validation_result["valid"] else RecordStatus.PENDING,
                metadata={
                    "created_from": "workflow_definition",
                    "validation_result": validation_result,
                    "disclaimers": disclaimers
                }
            )
            
            # Save to library
            await self.library.save_record(new_record)
            
            return {
                "status": "success",
                "action": "define_new",
                "record_id": new_record.id,
                "name": name,
                "validation": validation_result,
                "message": f"Defined new {item_type} {name}"
            }
    
    async def _execute_create_step(
        self, 
        step: Dict[str, Any], 
        domain: str, 
        firmware_content: str
    ) -> Dict[str, Any]:
        """
        Execute a CREATE step, which instantiates an agent or tool.
        
        Args:
            step: Step configuration
            domain: Domain for the step
            firmware_content: Firmware content for the domain
            
        Returns:
            Dictionary with step results
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        
        # Find the record
        record = await self.library.find_record_by_name(name)
        
        if not record:
            return {"status": "error", "message": f"Item not found: {name}"}
            
        if record.record_type != RecordType(item_type):
            return {"status": "error", "message": f"Item {name} is not a {item_type}"}
            
        if record.status != RecordStatus.ACTIVE:
            return {
                "status": "error", 
                "message": f"Item {name} is not active (status: {record.status.value})"
            }
        
        try:
            # Instantiate the agent or tool
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
                "action": "create",
                "record_id": record.id,
                "name": name,
                "message": f"Created {item_type} instance {name}"
            }
        except Exception as e:
            logger.error(f"Error creating {item_type} {name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error creating {item_type} {name}: {str(e)}"
            }
    
    async def _execute_execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an EXECUTE step, which runs an agent or tool.
        
        Args:
            step: Step configuration
            
        Returns:
            Dictionary with step results
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        user_input = step.get("user_input", "")
        
        # Check if the item is active
        if name not in self.active_items:
            return {"status": "error", "message": f"{item_type} {name} is not active"}
            
        active_item = self.active_items[name]
        if active_item["type"] != item_type:
            return {"status": "error", "message": f"Active item {name} is not a {item_type}"}
        
        try:
            # Execute the agent or tool
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
                
            # Update usage statistics
            await self.library.update_usage_metrics(active_item["record"].id, success=True)
            
            return {
                "status": "success",
                "action": "execute",
                "record_id": active_item["record"].id,
                "name": name,
                "result": result,
                "message": f"Executed {item_type} {name}"
            }
        except Exception as e:
            logger.error(f"Error executing {item_type} {name}: {str(e)}")
            
            # Update usage statistics with failure
            await self.library.update_usage_metrics(active_item["record"].id, success=False)
            
            return {
                "status": "error",
                "message": f"Error executing {item_type} {name}: {str(e)}"
            }