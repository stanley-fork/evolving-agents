# examples/architect_zero_comprehensive_demo.py

import asyncio
import logging
import json
import os
import re
import datetime
from dotenv import load_dotenv

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample invoice for testing
SAMPLE_INVOICE = """
INVOICE #12345
Date: 2023-05-15
Vendor: TechSupplies Inc.
Address: 123 Tech Blvd, San Francisco, CA 94107

Bill To:
Acme Corporation
456 Business Ave
New York, NY 10001

Items:
1. Laptop Computer - $1,200.00 (2 units)
2. External Monitor - $300.00 (3 units)
3. Wireless Keyboard - $50.00 (5 units)

Subtotal: $2,950.00
Tax (8.5%): $250.75
Total Due: $3,200.75

Payment Terms: Net 30
Due Date: 2023-06-14

Thank you for your business!
"""

def clean_previous_files():
    """Remove previous files to start fresh."""
    files_to_remove = [
        "smart_library.json",
        "agent_bus.json",
        "architect_interaction.txt",
        "invoice_workflow.yaml",
        "workflow_execution_result.json",
        "detailed_invoice_analysis.txt",
        "direct_invoice_analysis.txt",
        "invoice_processor_id.txt",
        "component_analysis.json"  # New file for component compatibility analysis
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")

async def setup_library():
    """Set up some initial components in the smart library to show evolution."""
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)  # Pass LLM service to SmartLibrary
    
    # Create a basic document analyzer with capabilities
    basic_doc_analyzer = {
        "name": "BasicDocumentAnalyzer",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "A basic tool that analyzes documents to determine their type",
        "code_snippet": """
from typing import Dict, Any
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")

class BasicDocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
    \"\"\"A basic tool that analyzes documents to determine their type.\"\"\"
    name = "BasicDocumentAnalyzer"
    description = "Analyzes document content to determine if it's an invoice, receipt, or other document type"
    input_schema = DocumentAnalyzerInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "document", "analyzer"],
            creator=self,
        )
    
    async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Analyze document text to determine its type.\"\"\"
        doc_text = input.text.lower()
        
        # Simple keyword matching
        result = {"type": "unknown", "confidence": 0.0}
        
        if "invoice" in doc_text and ("total" in doc_text or "amount" in doc_text):
            result = {"type": "invoice", "confidence": 0.7}
        elif "receipt" in doc_text:
            result = {"type": "receipt", "confidence": 0.6}
        elif "contract" in doc_text:
            result = {"type": "contract", "confidence": 0.6}
        
        import json
        return StringToolOutput(json.dumps(result, indent=2))
""",
        "version": "1.0.0",
        "tags": ["document", "analysis", "basic"],
        # Add capabilities information
        "capabilities": [
            {
                "id": "document_type_detection",
                "name": "Document Type Detection",
                "description": "Detects the type of document from its content",
                "context": {
                    "required_fields": ["document_text"],
                    "produced_fields": ["document_type", "confidence_score"]
                }
            }
        ]
    }
    
    # Create a basic invoice processor with capabilities
    basic_invoice_processor = {
        "name": "BasicInvoiceProcessor",
        "record_type": "AGENT",
        "domain": "document_processing",
        "description": "A basic agent that processes invoice documents to extract information",
        "code_snippet": """
from typing import List, Dict, Any, Optional
import re

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.memory import TokenMemory
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool

class BasicInvoiceProcessorInitializer:
    \"\"\"
    A basic agent that processes invoice documents to extract information.
    It can extract simple data like invoice number, date, and total amount.
    \"\"\"
    
    @staticmethod
    def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
        \"\"\"Create and configure the invoice processor agent.\"\"\"
        # Use empty tools list if none provided
        if tools is None:
            tools = []
            
        # Define agent metadata
        meta = AgentMeta(
            name="BasicInvoiceProcessor",
            description=(
                "I am an invoice processing agent that can extract basic information from invoice documents "
                "including invoice number, date, vendor, and total amount."
            ),
            tools=tools
        )
        
        # Create the agent
        agent = ReActAgent(
            llm=llm,
            tools=tools,
            memory=TokenMemory(llm),
            meta=meta
        )
        
        return agent
        
    @staticmethod
    async def process_invoice(invoice_text: str) -> Dict[str, Any]:
        \"\"\"
        Process an invoice to extract key information.
        
        Args:
            invoice_text: The text of the invoice to process
            
        Returns:
            Extracted invoice information
        \"\"\"
        # Extract invoice number
        invoice_num_match = re.search(r'INVOICE #([\\w-]+)', invoice_text, re.IGNORECASE)
        invoice_num = invoice_num_match.group(1) if invoice_num_match else "Unknown"
        
        # Extract date
        date_match = re.search(r'Date:?\\s*([\\w\\d/-]+)', invoice_text, re.IGNORECASE)
        date = date_match.group(1).strip() if date_match else "Unknown"
        
        # Extract vendor
        vendor_match = re.search(r'Vendor:?\\s*([^\\n]+)', invoice_text, re.IGNORECASE)
        vendor = vendor_match.group(1).strip() if vendor_match else "Unknown"
        
        # Extract total
        total_match = re.search(r'Total\\s*(?:Due|Amount)?:?\\s*\\$?([\\d.,]+)', invoice_text, re.IGNORECASE)
        total = total_match.group(1).strip() if total_match else "Unknown"
        
        return {
            "invoice_number": invoice_num,
            "date": date,
            "vendor": vendor,
            "total": total
        }
""",
        "version": "1.0.0",
        "tags": ["invoice", "processing", "basic"],
        # Add capabilities information
        "capabilities": [
            {
                "id": "invoice_data_extraction",
                "name": "Invoice Data Extraction",
                "description": "Extracts basic information from invoice documents",
                "context": {
                    "required_fields": ["invoice_text"],
                    "produced_fields": ["invoice_number", "date", "vendor", "total"]
                }
            }
        ]
    }
    
    # Add them to the library
    await smart_library.create_record(**basic_doc_analyzer)
    await smart_library.create_record(**basic_invoice_processor)
    
    logger.info("Set up initial components in the smart library")

async def main():
    # Clean up previous files
    clean_previous_files()
    
    # First, set up some initial components in the smart library
    await setup_library()
    
    # Initialize core components
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)  # Pass LLM service to SmartLibrary
    agent_bus = SimpleAgentBus("agent_bus.json")
    
    # Create the system agent
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus
    )

    system_agent.workflow_processor.set_llm_service(llm_service)
    system_agent.workflow_generator.set_llm_service(llm_service)
    
    # Create the Architect-Zero agent
    architect_agent = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent_factory=SystemAgentFactory.create_agent
    )
    
    # Define invoice processing task requirements
    task_requirement = """
    Create an advanced invoice processing system that improves upon the basic version in the library. The system should:
    
    1. Use a more sophisticated document analyzer that can detect invoices with higher confidence
    2. Extract comprehensive information (invoice number, date, vendor, items, subtotal, tax, total)
    3. Verify calculations to ensure subtotal + tax = total
    4. Generate a structured summary with key insights
    5. Handle different invoice formats and detect potential errors
    
    The system should leverage existing components from the library when possible,
    evolve them where improvements are needed, and create new components for missing functionality.
    
    Please generate a complete workflow for this invoice processing system.
    """
    
    # Print the task
    logger.info("=== TASK REQUIREMENTS ===")
    logger.info(task_requirement)
    
    # Use LLM-enhanced SmartLibrary to analyze required capabilities
    logger.info("\n=== ANALYZING WORKFLOW REQUIREMENTS WITH LLM ===")
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(task_requirement, "document_processing")
    logger.info(f"Extracted capabilities: {extracted_capabilities}")
    
    # Use enhanced find_components_for_workflow to get suggested components
    workflow_components = await smart_library.find_components_for_workflow(
        workflow_description=task_requirement,
        required_capabilities=extracted_capabilities,
        domain="document_processing",
        use_llm=True
    )
    
    logger.info(f"Identified components for workflow: {len(workflow_components)} capability matches")
    for cap_id, components in workflow_components.items():
        component_names = [c["name"] for c in components]
        logger.info(f"  - {cap_id}: {', '.join(component_names)}")
    
    # Run the architect agent to design the system
    logger.info("\n=== RUNNING ARCHITECT-ZERO AGENT ===")
    try:
        # Execute the architect agent as a ReAct agent
        result = await architect_agent.run(task_requirement)
        
        # Save the full agent interaction
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n{task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        logger.info("Architect-Zero completed successfully - see 'architect_interaction.txt' for full output")
        
        # Extract workflow from the result
        yaml_content = extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Save the workflow to a file
            with open("invoice_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            logger.info("Generated workflow saved to invoice_workflow.yaml")
        else:
            logger.warning("No YAML workflow found in the result")
            # Ask the LLM to generate one based on the response
            yaml_workflow = await generate_workflow_from_response(llm_service, result.result.text)
            if yaml_workflow:
                with open("invoice_workflow.yaml", "w") as f:
                    f.write(yaml_workflow)
                logger.info("Generated workflow saved to invoice_workflow.yaml")
        
        # Analyze component compatibility if we have a workflow
        if os.path.exists("invoice_workflow.yaml"):
            with open("invoice_workflow.yaml", "r") as f:
                yaml_content = f.read()
            
            # Extract component names from the workflow
            component_names = re.findall(r'name:\s*"?([^"\n]+)"?', yaml_content)
            
            # Create a dictionary of selected components
            selected_components = {}
            for name in component_names:
                for records in workflow_components.values():
                    for record in records:
                        if record["name"] == name:
                            for cap in record.get("capabilities", []):
                                cap_id = cap.get("id", "")
                                if cap_id:
                                    selected_components[cap_id] = record
                                    break
            
            # Analyze component compatibility
            if selected_components:
                logger.info("\n=== ANALYZING COMPONENT COMPATIBILITY ===")
                compatibility = await smart_library.analyze_workflow_component_compatibility(
                    workflow_description=task_requirement,
                    selected_components=selected_components
                )
                
                # Save compatibility analysis
                with open("component_analysis.json", "w") as f:
                    json.dump(compatibility, f, indent=2)
                
                logger.info(f"Component compatibility score: {compatibility.get('compatibility_score', 'unknown')}")
                if "missing_capabilities" in compatibility and compatibility["missing_capabilities"]:
                    logger.info(f"Missing capabilities: {', '.join(compatibility['missing_capabilities'])}")
                if "recommendations" in compatibility and compatibility["recommendations"]:
                    logger.info("Recommendations:")
                    for rec in compatibility["recommendations"]:
                        logger.info(f"  - {rec}")
        
        # Try to execute the workflow with sample invoice data
        logger.info("\n=== EXECUTING GENERATED WORKFLOW ===")
        execution_result = None
        try:
            # If we have a workflow, try to execute it
            if os.path.exists("invoice_workflow.yaml"):
                with open("invoice_workflow.yaml", "r") as f:
                    yaml_content = f.read()
                
                # Check if the workflow already has the invoice content with proper formatting
                if "INVOICE #12345" in yaml_content:
                    # Make sure the invoice is properly formatted as a YAML multi-line string
                    lines = yaml_content.split('\n')
                    new_lines = []
                    in_invoice = False
                    proper_indent = ""
                    
                    for i, line in enumerate(lines):
                        if "INVOICE #12345" in line and i > 0:
                            # Found the invoice start, check if previous line has user_input: |
                            if not lines[i-1].strip().endswith("user_input: |"):
                                # Need to fix formatting
                                in_invoice = True
                                # Get indentation from previous line
                                prev_line = lines[i-1]
                                proper_indent = prev_line[:prev_line.find("user_input:")] + "  "  # Add 2 spaces
                                # Replace previous line with proper formatting
                                new_lines[-1] = prev_line.replace("user_input:", "user_input: |")
                                # Add this line with proper indentation
                                new_lines.append(f"{proper_indent}{line.strip()}")
                            else:
                                # Already properly formatted
                                new_lines.append(line)
                        elif in_invoice:
                            # Continue indenting invoice lines
                            new_lines.append(f"{proper_indent}{line.strip()}")
                            # Check if we're at the end of the invoice
                            if not line.strip() or i == len(lines) - 1 or "type:" in lines[i+1]:
                                in_invoice = False
                        else:
                            # Regular line
                            new_lines.append(line)
                    
                    # Use the fixed content
                    yaml_content = '\n'.join(new_lines)
                    
                    # Write back the fixed content
                    with open("invoice_workflow.yaml", "w") as f:
                        f.write(yaml_content)
                
                # Use the workflow processor to execute it
                # Check available methods on workflow processor
                processor_methods = dir(system_agent.workflow_processor)
                logger.info(f"Available workflow processor methods: {[m for m in processor_methods if not m.startswith('_')]}")
                
                # Execute the workflow with the only parameter it accepts
                execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
                
                logger.info(f"Workflow execution result: {json.dumps(execution_result, indent=2)}")
                
                # Save execution result
                with open("workflow_execution_result.json", "w") as f:
                    json.dump(execution_result, f, indent=2)
                
                logger.info("Workflow execution result saved to workflow_execution_result.json")
            else:
                logger.warning("No workflow file found - skipping execution")
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Extract and display the actual processed invoice data if available
        try:
            if execution_result and "status" in execution_result and execution_result["status"] == "success":
                logger.info("\n=== INVOICE PROCESSING RESULTS ===")
                
                # Try to find any invoice processor agent
                processor_candidates = []
                for record in smart_library.records:
                    if record["record_type"] == "AGENT" and "invoice" in record["name"].lower():
                        processor_candidates.append(record)
                        
                if processor_candidates:
                    # Sort by creation date (newest first)
                    processor_candidates.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                    processor_agent_record = processor_candidates[0]
                    
                    # Save the agent ID for potential reuse
                    with open("invoice_processor_id.txt", "w") as f:
                        f.write(processor_agent_record["id"])
                    
                    logger.info(f"\n=== RUNNING {processor_agent_record['name']} DIRECTLY ===")
                    
                    try:
                        # Create the processor agent
                        from evolving_agents.agents.agent_factory import AgentFactory
                        agent_factory = AgentFactory(smart_library, llm_service, None)
                        
                        processor_agent = await agent_factory.create_agent(processor_agent_record)
                        
                        # Check if the agent was created successfully
                        if processor_agent:
                            # Use the LLM-enhanced query for better instruction formulation
                            enhanced_prompt = await smart_library._enhance_query_with_llm(
                                f"Process this invoice and provide a detailed analysis with extracted data, verification results, and a structured summary:\n\n{SAMPLE_INVOICE}",
                                "document_processing"
                            )
                            
                            # Use the run method directly with enhanced prompt
                            run_result = await processor_agent.run(prompt=enhanced_prompt)
                            
                            # Extract the result
                            if hasattr(run_result, "result") and hasattr(run_result.result, "text"):
                                result_text = run_result.result.text
                                logger.info("Direct execution result:")
                                logger.info(result_text)
                                
                                # Save detailed result
                                with open("detailed_invoice_analysis.txt", "w") as f:
                                    f.write(result_text)
                                
                                logger.info("Detailed invoice analysis saved to detailed_invoice_analysis.txt")
                            else:
                                logger.error(f"Unexpected result structure: {run_result}")
                    except Exception as e:
                        logger.error(f"Error running agent directly: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    # If no invoice processor found, log this
                    logger.warning("No invoice processor agent found in library")
                
                # Generate a final invoice analysis using the LLM directly
                logger.info("\n=== GENERATING INVOICE ANALYSIS DIRECTLY ===")
                analysis_prompt = f"""
                Analyze this invoice and provide a detailed breakdown:
                
                {SAMPLE_INVOICE}
                
                Please provide:
                1. Extracted key information (invoice #, date, vendor, etc.)
                2. Line items with quantities and amounts
                3. Verification of calculations (subtotal + tax = total)
                4. Any potential errors or inconsistencies
                5. A structured summary with insights
                
                Format your response as a clear, structured analysis.
                """
                
                direct_analysis = await llm_service.generate(analysis_prompt)
                
                # Save direct analysis
                with open("direct_invoice_analysis.txt", "w") as f:
                    f.write(direct_analysis)
                    
                logger.info("Generated direct invoice analysis and saved to direct_invoice_analysis.txt")
                
                # Track performance metrics in the library for the components that were used
                if "records" in execution_result:
                    for record in execution_result["records"]:
                        component_id = record.get("id")
                        success = record.get("status") == "success"
                        if component_id:
                            # Update performance metrics
                            await smart_library.update_component_metrics(
                                component_id=component_id,
                                execution_metrics={
                                    "success": success,
                                    "execution_time": record.get("execution_time", 1.0),
                                    "capability_id": "invoice_processing",  # Generic capability
                                    "domain": "document_processing",
                                    "timestamp": datetime.datetime.utcnow().isoformat()
                                }
                            )
                            logger.info(f"Updated performance metrics for component {component_id}")
        except Exception as e:
            logger.error(f"Error getting detailed invoice results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error running Architect-Zero: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def extract_yaml_workflow(text):
    """Extract YAML workflow from the agent's response."""
    # Try to extract code between ```yaml and ``` markers
    yaml_match = re.search(r'```yaml\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1).strip()
    else:
        # Try with different syntax
        yaml_match2 = re.search(r'```\s*\n(scenario_name:.*?)\n\s*```', text, re.DOTALL)
        if yaml_match2:
            yaml_content = yaml_match2.group(1).strip()
        else:
            # Try to extract code between yaml: and any other markdown section
            yaml_section_match = re.search(r'yaml_workflow["\']?:\s*["\']?(.*?)(?:(?=["\']?,\s*["\']?\w+["\']?:)|$)', 
                                         text, re.DOTALL)
            if yaml_section_match:
                content = yaml_section_match.group(1).strip()
                # Remove any trailing quotes
                if content.endswith('"') or content.endswith("'"):
                    content = content[:-1]
                # Remove any leading quotes
                if content.startswith('"') or content.startswith("'"):
                    content = content[1:]
                yaml_content = content
            else:
                # More fallbacks...
                yaml_section = None
                if "YAML Workflow:" in text:
                    parts = text.split("YAML Workflow:")
                    if len(parts) > 1:
                        yaml_section = parts[1].strip()
                        # Find where it ends (next heading or end of text)
                        heading_match = re.search(r'\n#', yaml_section)
                        if heading_match:
                            yaml_section = yaml_section[:heading_match.start()].strip()
                
                if yaml_section and yaml_section.strip().startswith('scenario_name:'):
                    yaml_content = yaml_section
                else:
                    # Look for YAML content without a specific header
                    lines = text.split('\n')
                    yaml_lines = []
                    collecting = False
                    
                    for line in lines:
                        if not collecting and line.strip().startswith('scenario_name:'):
                            collecting = True
                            yaml_lines.append(line)
                        elif collecting:
                            if line.strip().startswith('#') or line.strip().startswith('```'):
                                break
                            yaml_lines.append(line)
                    
                    if yaml_lines:
                        yaml_content = '\n'.join(yaml_lines)
                    else:
                        return None
    
    # Check if the YAML includes an EXECUTE step with user_input
    if "user_input:" in yaml_content and not "INVOICE #" in yaml_content:
        # Replace the user_input with our sample invoice
        lines = yaml_content.split('\n')
        for i, line in enumerate(lines):
            if "user_input:" in line and not "user_input: |" in line:
                # Fix: Ensure proper YAML formatting for multi-line string
                indent = line[:line.index("user_input:")]
                # Replace with pipe notation for multi-line strings
                lines[i] = f"{indent}user_input: |"
                # Add the sample invoice with proper indentation
                indent_level = len(indent) + 2  # Add 2 spaces for the sub-indentation
                invoice_indent = " " * indent_level
                for invoice_line in SAMPLE_INVOICE.strip().split('\n'):
                    lines.insert(i+1, f"{invoice_indent}{invoice_line}")
                break
        
        yaml_content = '\n'.join(lines)
    
    return yaml_content

async def generate_workflow_from_response(llm_service, full_text):
    """Use the LLM to extract or generate a YAML workflow from the response."""
    prompt = f"""
    Create a complete YAML workflow for an advanced invoice processing system.
    
    The workflow MUST include:
    1. A definition for an "AdvancedInvoiceProcessor" agent as the main orchestrator
    2. Definitions for specialized component agents (not tools) for different invoice processing tasks
    3. Creation steps for all defined components
    4. Execution of the AdvancedInvoiceProcessor agent with the sample invoice
    
    REQUIRED COMPONENTS:
    - AdvancedInvoiceProcessor (AGENT - main orchestrator)
    - DocumentAnalyzer (AGENT)
    - DataExtractor (AGENT)
    - CalculationVerifier (AGENT)
    - SummaryGenerator (AGENT)
    - ErrorDetector (AGENT)
    
    SAMPLE INVOICE:
    {SAMPLE_INVOICE}

    YAML FORMAT:
    ```yaml
    scenario_name: Advanced Invoice Processing System
    domain: document_processing
    description: A system to process invoice documents with advanced capabilities

    steps:
      # Define the AdvancedInvoiceProcessor orchestrator agent (REQUIRED)
      - type: "DEFINE"
        item_type: "AGENT"
        name: "AdvancedInvoiceProcessor"
        description: "Main agent that orchestrates the invoice processing workflow"
        code_snippet: |
          from typing import List, Dict, Any, Optional
          import re

          from beeai_framework.agents.react import ReActAgent
          from beeai_framework.agents.types import AgentMeta
          from beeai_framework.memory import TokenMemory
          from beeai_framework.backend.chat import ChatModel
          from beeai_framework.tools.tool import Tool

          class AdvancedInvoiceProcessorInitializer:
              \"\"\"
              Advanced invoice processor that orchestrates specialized components
              to analyze, extract, verify, and summarize invoice information.
              \"\"\"
              
              @staticmethod
              def create_agent(llm: ChatModel, tools: Optional[List[Tool]] = None) -> ReActAgent:
                  \"\"\"Create and configure the invoice processor agent.\"\"\"
                  # Use empty tools list if none provided
                  if tools is None:
                      tools = []
                      
                  # Define agent metadata
                  meta = AgentMeta(
                      name="AdvancedInvoiceProcessor",
                      description=(
                          "I am an advanced invoice processing agent that orchestrates specialized components "
                          "to analyze, extract, verify, and summarize invoice information."
                      ),
                      tools=tools
                  )
                  
                  # Create the agent
                  agent = ReActAgent(
                      llm=llm,
                      tools=tools,
                      memory=TokenMemory(llm),
                      meta=meta
                  )
                  
                  return agent
          
      # Define other component agents
      - type: "DEFINE"
        item_type: "AGENT"
        name: "DocumentAnalyzer"
        description: "Agent that analyzes document structure and identifies invoice types"
        code_snippet: |
          # Agent implementation code
          
      # ... other component definitions
      
      # Create all agents (REQUIRED)
      - type: "CREATE"
        item_type: "AGENT"
        name: "AdvancedInvoiceProcessor"
      
      # ... other agent creations
      
      # Execute the AdvancedInvoiceProcessor (REQUIRED)
      - type: "EXECUTE"
        item_type: "AGENT"
        name: "AdvancedInvoiceProcessor"
        user_input: |
          Process this invoice:
          
          {SAMPLE_INVOICE}
    ```

    Return only the YAML workflow without explanation.
    """
    
    response = await llm_service.generate(prompt)
    
    # Extract the YAML
    yaml_match = re.search(r'```yaml\s*\n(.*?)\n\s*```', response, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1).strip()
    
    # If not found with yaml marker, try without specific language
    yaml_match2 = re.search(r'```\s*\n(scenario_name:.*?)\n\s*```', response, re.DOTALL)
    if yaml_match2:
        return yaml_match2.group(1).strip()
    
    # If still not found, return the full response if it looks like YAML
    if response.strip().startswith('scenario_name:'):
        return response.strip()
    
    return None

if __name__ == "__main__":
    asyncio.run(main())