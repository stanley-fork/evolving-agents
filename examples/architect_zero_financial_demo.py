import asyncio
import logging
import json
import os
import re
import time
import colorama
from colorama import Fore, Style

from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper function for pretty printing
def print_step(title, content=None, step_type="INFO"):
    """Print a beautifully formatted step."""
    colors = {
        "INFO": Fore.BLUE,
        "AGENT": Fore.GREEN,
        "REASONING": Fore.YELLOW,
        "EXECUTION": Fore.CYAN,
        "SUCCESS": Fore.GREEN,
        "ERROR": Fore.RED
    }
    
    color = colors.get(step_type, Fore.WHITE)
    
    # Print header
    print(f"\n{color}{'=' * 80}")
    print(f"  {step_type}: {title}")
    print(f"{'=' * 80}{Style.RESET_ALL}")
    
    # Print content if provided
    if content:
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"{Fore.CYAN}{key}{Style.RESET_ALL}: {value}")
        else:
            print(content)

# Sample invoice for demonstration
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
        "component_analysis.json"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")

async def setup_library():
    """Set up some initial components in the smart library."""
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)
    
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

async def extract_yaml_workflow(text):
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
    
    # Process the YAML content to include the sample invoice
    if "user_input:" in yaml_content and "INVOICE #12345" not in yaml_content:
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

async def main():
    print_step("INVOICE PROCESSING WITH AUTONOMOUS AGENT COLLABORATION", 
              "This demonstration shows how a system of specialized agents can collaborate to process invoices with complete visibility into their reasoning.", 
              "INFO")
    
    # Clean up previous files
    clean_previous_files()
    
    # Set up initial components in the library
    print_step("INITIALIZING COMPONENT LIBRARY", 
              "Creating foundational components that our agents can discover and leverage.", 
              "INFO")
    await setup_library()
    
    # Initialize core components
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)
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
    print_step("CREATING ARCHITECT-ZERO META-AGENT", 
              "This agent designs and orchestrates specialized agent systems to solve complex tasks.", 
              "AGENT")
    architect_agent = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent=system_agent
    )
    
    
    # Define invoice processing task
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
    print_step("TASK REQUIREMENTS", task_requirement, "INFO")
    
    # Use LLM-enhanced SmartLibrary to analyze required capabilities
    print_step("ANALYZING TASK REQUIREMENTS", 
              "Architect-Zero is extracting required capabilities from the task description.", 
              "REASONING")
    
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(task_requirement, "document_processing")
    print_step("CAPABILITY EXTRACTION RESULTS", {
        "Extracted capabilities": ", ".join(extracted_capabilities)
    }, "REASONING")
    
    # Find components that match these capabilities
    print_step("DISCOVERING EXISTING COMPONENTS", 
              "Searching for components that can fulfill the required capabilities.", 
              "REASONING")
    
    workflow_components = await smart_library.find_components_for_workflow(
        workflow_description=task_requirement,
        required_capabilities=extracted_capabilities,
        domain="document_processing",
        use_llm=True
    )
    
    capability_matches = {}
    for cap_id, components in workflow_components.items():
        component_names = [c["name"] for c in components]
        capability_matches[cap_id] = ", ".join(component_names) if component_names else "No match found"
    
    print_step("COMPONENT DISCOVERY RESULTS", capability_matches, "REASONING")
    
    # Execute Architect-Zero to design the system
    print_step("DESIGNING INVOICE PROCESSING SYSTEM", 
              "Architect-Zero is designing a multi-agent solution with full reasoning visibility.", 
              "AGENT")
    
    try:
        # Execute the architect agent
        print(f"{Fore.GREEN}Starting agent reasoning process...{Style.RESET_ALL}")
        start_time = time.time()
        result = await architect_agent.run(task_requirement)
        design_time = time.time() - start_time
        
        # Save the full thought process
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n{task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        # Show the reasoning process (truncated)
        reasoning_preview = result.result.text[:500] + "..." if len(result.result.text) > 500 else result.result.text
        print_step("AGENT REASONING REVEALED", {
            "Design time": f"{design_time:.2f} seconds",
            "Reasoning preview": reasoning_preview,
            "Full reasoning": "Saved to 'architect_interaction.txt'"
        }, "REASONING")
        
        # Extract workflow from the result
        yaml_content = await extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Save the workflow to a file
            with open("invoice_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            print_step("GENERATED MULTI-AGENT WORKFLOW", 
                      "Architect-Zero has created a complete workflow of specialized agents.", 
                      "SUCCESS")
            
            # Show abbreviated workflow
            workflow_lines = yaml_content.split('\n')
            workflow_preview = '\n'.join(workflow_lines[:20])
            if len(workflow_lines) > 20:
                workflow_preview += f"\n{Fore.CYAN}... (see invoice_workflow.yaml for complete workflow){Style.RESET_ALL}"
            print(workflow_preview)
            
            # Execute the workflow
            print_step("EXECUTING MULTI-AGENT WORKFLOW", 
                      "Now watching the agents collaborate on processing the invoice.", 
                      "EXECUTION")
            
            workflow_start_time = time.time()
            execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
            workflow_time = time.time() - workflow_start_time
            
            # Save execution result
            with open("workflow_execution_result.json", "w") as f:
                json.dump(execution_result, f, indent=2)
            
            # Show execution results
            if execution_result and execution_result.get("status") == "success":
                print_step("WORKFLOW EXECUTION RESULTS", {
                    "Execution time": f"{workflow_time:.2f} seconds",
                    "Status": execution_result.get("status")
                }, "SUCCESS")
                
                # Extract the actual invoice processing results
                result_text = execution_result.get("result", "")
                print_step("AGENT COLLABORATION OUTPUT", 
                          "Each agent's contribution is visible in the complete execution trace:", 
                          "EXECUTION")
                
                # Process the result text to make it more readable
                # Try to extract each agent's contribution
                agent_results = re.findall(r'\*\*(.*?)\*\*.*?Outcome: (.*?)(?:\n\n|$)', result_text, re.DOTALL)
                
                if agent_results:
                    for agent, result in agent_results:
                        print(f"{Fore.GREEN}✓ {agent}{Style.RESET_ALL}: {result.strip()}")
                else:
                    # Fallback to showing the full text
                    print(result_text)
                    
                # Analyze the agent collaboration
                agent_count = len(re.findall(r'type: "CREATE"\s+item_type: "AGENT"', yaml_content))
                agent_dependencies = len(re.findall(r'user_input: \|', yaml_content))
                
                print_step("AGENT COLLABORATION INSIGHTS", {
                    "Number of specialized agents": agent_count,
                    "Inter-agent dependencies": agent_dependencies,
                    "Execution transparency": "Full visibility into each agent's reasoning and contribution",
                    "Key advantage": "Complete trace of how the system reached its conclusions"
                }, "SUCCESS")
            else:
                print_step("WORKFLOW EXECUTION ISSUE", 
                          f"Status: {execution_result.get('status', 'unknown')}, Message: {execution_result.get('message', 'Unknown error')}", 
                          "ERROR")
        else:
            print_step("WORKFLOW GENERATION ISSUE", 
                      "No YAML workflow found in the agent's response.", 
                      "ERROR")
            
    except Exception as e:
        print_step("ERROR", str(e), "ERROR")
        import traceback
        print(traceback.format_exc())
    
    print_step("DEMONSTRATION COMPLETED", 
              "This demonstration showed how specialized agents can collaborate with visible reasoning, providing transparency in AI decision-making.", 
              "INFO")

if __name__ == "__main__":
    asyncio.run(main())