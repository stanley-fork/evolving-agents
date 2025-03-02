# evolving_agents/workflow/workflow_processor.py

import logging
import yaml
import uuid
from typing import Dict, Any, List, Optional

from beeai_framework.tools.tool import Tool

from evolving_agents.core.system_agent import SystemAgent

logger = logging.getLogger(__name__)

class WorkflowProcessor:
    """
    Processes YAML-based workflows as described in Article 3.2.
    """
    def __init__(self, system_agent: SystemAgent):
        self.system_agent = system_agent
        logger.info("Workflow Processor initialized")
    
    async def process_workflow(self, workflow_yaml: str) -> Dict[str, Any]:
        """
        Process a workflow from a YAML string.
        
        Args:
            workflow_yaml: YAML string defining the workflow
            
        Returns:
            Execution results
        """
        logger.info("Processing workflow")
        
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
        firmware_content = self.system_agent.firmware.get_firmware_prompt(domain)
        
        # Execute steps
        results = {
            "scenario_name": scenario_name,
            "domain": domain,
            "steps": []
        }
        
        # Keep track of defined tools for agent creation
        defined_tools = {}
        
        for i, step in enumerate(workflow.get("steps", [])):
            step_type = step.get("type", "").upper()
            logger.info(f"Executing step {i+1}: {step_type}")
            
            try:
                if step_type == "DEFINE":
                    step_result = await self._process_define_step(step, domain, firmware_content, disclaimers)
                    
                    # Keep track of defined tools
                    if step.get("item_type") == "TOOL":
                        defined_tools[step.get("name")] = step_result.get("record_id")
                        
                elif step_type == "CREATE":
                    # If creating an agent, collect required tools
                    tools_to_use = []
                    if step.get("item_type") == "AGENT":
                        agent_name = step.get("name")
                        agent_record = await self.system_agent.library.find_record_by_name(agent_name)
                        
                        if agent_record and "required_tools" in agent_record.get("metadata", {}):
                            required_tool_names = agent_record["metadata"]["required_tools"]
                            for tool_name in required_tool_names:
                                if tool_name in self.system_agent.active_items:
                                    tools_to_use.append(self.system_agent.active_items[tool_name]["instance"])
                    
                    step_result = await self._process_create_step(step, domain, firmware_content, tools_to_use)
                    
                elif step_type == "EXECUTE":
                    step_result = await self._process_execute_step(step)
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
    
    async def _process_define_step(
        self,
        step: Dict[str, Any],
        domain: str,
        firmware_content: str,
        disclaimers: List[str]
    ) -> Dict[str, Any]:
        """
        Process a DEFINE step to define an agent or tool.
        
        Args:
            step: Step definition
            domain: Domain for the step
            firmware_content: Firmware content
            disclaimers: Additional disclaimers
            
        Returns:
            Step result
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        description = step.get("description", "")
        
        logger.info(f"Defining {item_type} '{name}'")
        
        # Check if we need to reuse or evolve an existing item
        if "from_existing_snippet" in step:
            source_name = step.get("from_existing_snippet")
            source_record = await self.system_agent.library.find_record_by_name(source_name)
            
            if not source_record:
                return {"status": "error", "message": f"Source item not found: {source_name}"}
            
            if step.get("reuse_as_is", False):
                # Simple reuse
                message = f"Reused {source_name} as {name}"
                logger.info(message)
                
                # Just copy with a new name
                new_record = source_record.copy()
                new_record["id"] = str(uuid.uuid4())
                new_record["name"] = name
                new_record["description"] = description or source_record["description"]
                new_record["metadata"] = source_record.get("metadata", {}).copy()
                new_record["metadata"]["reused_from"] = source_record["id"]
                new_record["metadata"]["disclaimers"] = disclaimers
                
                # Add required_tools if specified for agents
                if item_type == "AGENT" and "required_tools" in step:
                    new_record["metadata"]["required_tools"] = step["required_tools"]
                
                await self.system_agent.library.save_record(new_record)
                
                return {
                    "status": "success",
                    "action": "reuse",
                    "message": message,
                    "record_id": new_record["id"]
                }
            else:
                # Evolve
                evolve_changes = step.get("evolve_changes", {})
                docstring_update = evolve_changes.get("docstring_update", "")
                
                message = f"Evolved {source_name} to {name}, injecting firmware for domain '{domain}'."
                logger.info(message)
                
                # Generate evolved code
                evolve_prompt = f"""
                {firmware_content}
                
                ORIGINAL CODE:
                ```
                {source_record['code_snippet']}
                ```
                
                REQUIRED CHANGES:
                - New name: {name}
                - Description: {description or source_record['description']}
                - Docstring update: {docstring_update}
                
                DOMAIN REQUIREMENTS:
                - Domain: {domain}
                
                REQUIRED DISCLAIMERS:
                {chr(10).join(disclaimers)}
                
                Evolve the code to implement these changes while maintaining core functionality.
                Include all required disclaimers and follow domain guidelines.
                
                EVOLVED CODE:
                """
                
                evolved_code = await self.system_agent.llm.generate(evolve_prompt)
                
                # Create evolved record
                metadata = {
                    "evolved_from": source_record["id"],
                    "evolution_changes": evolve_changes,
                    "disclaimers": disclaimers
                }
                
                # Add required_tools if specified for agents
                if item_type == "AGENT" and "required_tools" in step:
                    metadata["required_tools"] = step["required_tools"]
                
                new_record = await self.system_agent.library.create_record(
                    name=name,
                    record_type=item_type,
                    domain=domain,
                    description=description or source_record["description"],
                    code_snippet=evolved_code,
                    metadata=metadata
                )
                
                return {
                    "status": "success",
                    "action": "evolve",
                    "message": message,
                    "record_id": new_record["id"]
                }
        else:
            # Create new from scratch
            message = f"Created new {item_type} '{name}' from scratch"
            logger.info(message)
            
            # Generate code
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
            
            new_code = await self.system_agent.llm.generate(create_prompt)
            
            # Prepare metadata
            metadata = {
                "created_from": "workflow",
                "disclaimers": disclaimers
            }
            
            # Add required_tools if specified for agents
            if item_type == "AGENT" and "required_tools" in step:
                metadata["required_tools"] = step["required_tools"]
            
            # Create record
            new_record = await self.system_agent.library.create_record(
                name=name,
                record_type=item_type,
                domain=domain,
                description=description,
                code_snippet=new_code,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "action": "create",
                "message": message,
                "record_id": new_record["id"]
            }
    
    async def _process_create_step(
        self, 
        step: Dict[str, Any], 
        domain: str, 
        firmware_content: str,
        tools: Optional[List[Tool]] = None
    ) -> Dict[str, Any]:
        """
        Process a CREATE step to instantiate an agent or tool.
        
        Args:
            step: Step definition
            domain: Domain for the step
            firmware_content: Firmware content
            tools: Optional tools to provide to agents
            
        Returns:
            Step result
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        
        message = f"Created {item_type} instance {name}"
        logger.info(message)
        
        # Find the record
        record = await self.system_agent.library.find_record_by_name(name)
        
        if not record:
            return {"status": "error", "message": f"Item not found: {name}"}
        
        if record["record_type"] != item_type:
            return {"status": "error", "message": f"Item {name} is not a {item_type}"}
        
        # Create instance based on record type
        try:
            if item_type == "AGENT":
                instance = await self.system_agent.agent_factory.create_agent(
                    record=record,
                    firmware_content=firmware_content,
                    tools=tools
                )
            else:  # TOOL
                instance = await self.system_agent.tool_factory.create_tool(
                    record=record,
                    firmware_content=firmware_content
                )
            
            # Store in active items
            self.system_agent.active_items[name] = {
                "record": record,
                "instance": instance,
                "type": record["record_type"]
            }
            
            return {
                "status": "success",
                "message": message,
                "record_id": record["id"]
            }
        except Exception as e:
            logger.error(f"Error creating {item_type} {name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error creating {item_type} {name}: {str(e)}"
            }
    
    async def _process_execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an EXECUTE step to run an agent or tool.
        
        Args:
            step: Step definition
            
        Returns:
            Step result
        """
        item_type = step.get("item_type", "").upper()
        name = step.get("name")
        user_input = step.get("user_input", "")
        
        message = f"Executed {item_type} {name}"
        logger.info(message)
        
        # Execute using system agent's execute_item method
        result = await self.system_agent.execute_item(name, user_input)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": message,
                "result": result["result"],
                "record_id": self.system_agent.active_items.get(name, {}).get("record", {}).get("id")
            }
        else:
            return {
                "status": "error",
                "message": result["message"]
            }