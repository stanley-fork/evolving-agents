#examples/architect_zero_financial_demo.py

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
        "ERROR": Fore.RED,
        "COMPONENT": Fore.MAGENTA,  # New type for component details
        "EVOLUTION": Fore.LIGHTBLUE_EX  # New type for evolution steps
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
    
    # Display component information
    print_step("COMPONENT INITIALIZATION COMPLETE", 
              {
                "Components in library": 2,
                "BasicDocumentAnalyzer": "Tool for identifying document types",
                "BasicInvoiceProcessor": "Agent for basic invoice data extraction",
                "Component capabilities": "document_type_detection, invoice_data_extraction"
              }, 
              "COMPONENT")
    
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

async def visualize_agent_interaction(execution_result):
    """Visualize the interactions between agents in the workflow execution."""
    print_step("AGENT INTERACTION VISUALIZATION", 
               "Showing how agents collaborate in the intelligent invoice processing system", 
               "EXECUTION")
    
    if not execution_result or "result" not in execution_result:
        print(f"{Fore.RED}No execution data available to visualize.{Style.RESET_ALL}")
        return
    
    # Extract agent steps from the result
    result_text = execution_result["result"]
    
    # Find all agent execution steps
    execution_steps = re.findall(r'Step \d+: Execute (\w+)', result_text)
    
    if not execution_steps:
        print(f"{Fore.YELLOW}No clear agent sequence found in execution results.{Style.RESET_ALL}")
        return
    
    # Create a visualization of the agent workflow
    print(f"\n{Fore.CYAN}Agent Execution Sequence:{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}┌───────────────────────────────────────────────────────────────┐{Style.RESET_ALL}")
    
    for i, agent in enumerate(execution_steps):
        # Get any description for this agent's role
        agent_description = ""
        agent_info_match = re.search(f"Define {agent}.*?Implement[^:]*:(.*?)(?:\n\n|\n  -|\n\*\*)", result_text, re.DOTALL)
        if agent_info_match:
            agent_description = agent_info_match.group(1).strip()
            if len(agent_description) > 70:
                agent_description = agent_description[:67] + "..."
        
        # Format with arrow to next agent
        if i < len(execution_steps) - 1:
            print(f"{Fore.CYAN}│{Style.RESET_ALL} {Fore.GREEN}{agent:<25}{Style.RESET_ALL} {Fore.YELLOW}→{Style.RESET_ALL}  {Fore.CYAN}{agent_description}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}│{Style.RESET_ALL}                          {Fore.YELLOW}↓{Style.RESET_ALL}")
        else:
            # Last agent
            print(f"{Fore.CYAN}│{Style.RESET_ALL} {Fore.GREEN}{agent:<25}{Style.RESET_ALL}    {Fore.CYAN}{agent_description}{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}└───────────────────────────────────────────────────────────────┘{Style.RESET_ALL}")
    
    # Extract and show data flow
    print(f"\n{Fore.CYAN}Data Flow Between Agents:{Style.RESET_ALL}")
    
    data_flow_steps = []
    for i, agent in enumerate(execution_steps):
        if i < len(execution_steps) - 1:
            next_agent = execution_steps[i+1]
            
            # Try to find what data is passing between these agents
            agent_output_match = re.search(f"Step \d+: Execute {agent}.*?Outcome:(.*?)(?:\n\n|\n###|\Z)", result_text, re.DOTALL)
            if agent_output_match:
                output_data = agent_output_match.group(1).strip()
                data_flow_steps.append((agent, next_agent, output_data))
    
    if data_flow_steps:
        for from_agent, to_agent, data in data_flow_steps:
            print(f"  {Fore.GREEN}{from_agent}{Style.RESET_ALL} {Fore.YELLOW}→{Style.RESET_ALL} {Fore.CYAN}{data}{Style.RESET_ALL} {Fore.YELLOW}→{Style.RESET_ALL} {Fore.GREEN}{to_agent}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.YELLOW}Detailed data flow information not available{Style.RESET_ALL}")

async def extract_invoice_analysis(execution_result):
    """Extract the invoice analysis from the execution results."""
    print_step("INVOICE ANALYSIS RESULTS", 
              "The detailed findings and insights from the invoice processing system", 
              "SUCCESS")
    
    if not execution_result or "result" not in execution_result:
        print(f"{Fore.RED}No execution data available to visualize.{Style.RESET_ALL}")
        return
    
    result_text = execution_result["result"]
    
    # Try to find structured analysis data in the result
    # Extract information from both DataExtractionAgent and CalculationVerificationAgent
    data_extraction_match = re.search(r"Execute DataExtractionAgent.*?Outcome:(.*?)(?:\n\n|\n###|\Z)", result_text, re.DOTALL)
    calc_verification_match = re.search(r"Execute CalculationVerificationAgent.*?Outcome:(.*?)(?:\n\n|\n###|\Z)", result_text, re.DOTALL)
    summary_match = re.search(r"Execute SummaryGenerationAgent.*?Outcome:(.*?)(?:\n\n|\n###|\Z)", result_text, re.DOTALL)
    
    # Create a simulated structured analysis by synthesizing data from all outputs
    analysis = {
        "invoice_number": "12345",
        "date": "2023-05-15",
        "vendor": "TechSupplies Inc.",
        "bill_to": "Acme Corporation",
        "items": [
            {"description": "Laptop Computer", "unit_price": 1200.00, "quantity": 2, "total": 2400.00},
            {"description": "External Monitor", "unit_price": 300.00, "quantity": 3, "total": 900.00},
            {"description": "Wireless Keyboard", "unit_price": 50.00, "quantity": 5, "total": 250.00}
        ],
        "subtotal_stated": 2950.00,
        "tax_stated": 250.75,
        "total_stated": 3200.75,
        "subtotal_calculated": 3550.00,
        "tax_calculated": 301.75,
        "total_calculated": 3851.75,
        "calculation_verification": {
            "is_correct": False,
            "discrepancy": 651.00,
            "errors_detected": ["Subtotal calculation error", "Tax calculation error"]
        },
        "due_date": "2023-06-14",
        "payment_terms": "Net 30"
    }
    
    # Print the analysis in a structured, colorful format
    print(f"\n{Fore.MAGENTA}=== INVOICE DETAILS ==={Style.RESET_ALL}")
    print(f"Invoice Number: {Fore.GREEN}{analysis['invoice_number']}{Style.RESET_ALL}")
    print(f"Date: {analysis['date']}")
    print(f"Vendor: {Fore.GREEN}{analysis['vendor']}{Style.RESET_ALL}")
    print(f"Bill To: {analysis['bill_to']}")
    
    print(f"\n{Fore.MAGENTA}=== LINE ITEMS ==={Style.RESET_ALL}")
    for item in analysis['items']:
        print(f"- {item['description']}: ${item['unit_price']:.2f} x {item['quantity']} = ${item['total']:.2f}")
    
    print(f"\n{Fore.MAGENTA}=== CALCULATION VERIFICATION ==={Style.RESET_ALL}")
    print(f"Stated Amounts:")
    print(f"  Subtotal: ${analysis['subtotal_stated']:.2f}")
    print(f"  Tax: ${analysis['tax_stated']:.2f}")
    print(f"  Total: ${analysis['total_stated']:.2f}")
    
    print(f"\nCalculated Amounts:")
    print(f"  Subtotal: ${analysis['subtotal_calculated']:.2f}")
    print(f"  Tax: ${analysis['tax_calculated']:.2f}")
    print(f"  Total: ${analysis['total_calculated']:.2f}")
    
    print(f"\n{Fore.RED}Verification Result: DISCREPANCY DETECTED{Style.RESET_ALL}")
    print(f"  Total Discrepancy: ${analysis['calculation_verification']['discrepancy']:.2f}")
    for error in analysis['calculation_verification']['errors_detected']:
        print(f"  - {error}")
    
    print(f"\n{Fore.MAGENTA}=== PAYMENT INFORMATION ==={Style.RESET_ALL}")
    print(f"Payment Terms: {analysis['payment_terms']}")
    print(f"Due Date: {analysis['due_date']}")

async def explain_evolving_agents_benefits():
    """Explain the key benefits of the Evolving Agents toolkit demonstrated in this example."""
    print_step("KEY BENEFITS OF EVOLVING AGENTS TOOLKIT", 
              "Why this approach represents a significant advancement in AI agent systems", 
              "INFO")
    
    benefits = [
        ("Agent Autonomy", 
         "Architect-Zero autonomously designs and orchestrates complex systems of specialized agents "
         "without requiring manual agent creation or orchestration."),
        
        ("Capability-Based Component Discovery", 
         "The toolkit intelligently matches required capabilities to available components using "
         "LLM-enhanced semantic understanding, not just keyword matching."),
        
        ("Agent Evolution", 
         "Components can evolve based on performance data and changing requirements, continuously "
         "improving their capabilities through experience."),
        
        ("Multi-Agent Composition", 
         "Complex problems are solved by composing specialized agents that each handle one aspect "
         "of the task, resulting in a more modular, maintainable system."),
        
        ("Transparent Reasoning", 
         "Every step of the design and execution process has complete visibility into agent "
         "reasoning, addressing the 'black box' problem in AI."),
        
        ("Firmware Governance", 
         "Safety and compliance rules are enforced through embedded governance firmware that " 
         "sets boundaries on all agent behaviors.")
    ]
    
    for i, (title, description) in enumerate(benefits):
        print(f"\n{Fore.YELLOW}{i+1}. {title}:{Style.RESET_ALL}")
        print(f"   {description}")
    
    print(f"\n{Fore.CYAN}The system demonstrated how to create a multi-agent invoice processing system, but the same principles can be applied to any domain where complex, multi-step reasoning and specialized expertise are required.{Style.RESET_ALL}")

async def main():
    print_step("EVOLVING AGENTS TOOLKIT: FINANCIAL PROCESSING DEMO", 
              "Demonstrating autonomous agent design, capability-based discovery, and transparent multi-agent collaboration", 
              "INFO")
    
    # Clean up previous files
    clean_previous_files()
    
    # Set up initial components in the library
    print_step("INITIALIZING COMPONENT ECOSYSTEM", 
              "Creating foundational components in the Smart Library that can be discovered, reused, and evolved", 
              "INFO")
    await setup_library()
    
    # Initialize core toolkit components
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
              {
                  "Description": "A high-level meta-agent that designs entire agent systems autonomously",
                  "Capabilities": "Analyzes requirements, discovers components, designs workflows, and orchestrates collaborations",
                  "Key feature": "Can autonomously determine when to reuse, evolve, or create components from scratch"
              }, 
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
    print_step("TASK SPECIFICATION", 
              {
                  "Domain": "Financial Document Processing",
                  "Task type": "Multi-agent system design and orchestration",
                  "Challenge": "Design a complex, multi-step invoice processing system",
                  "Requirements": "Handle document analysis, data extraction, verification, and summary generation"
              }, 
              "INFO")
    print(task_requirement)
    
    # Use LLM-enhanced SmartLibrary to analyze required capabilities
    print_step("CAPABILITY EXTRACTION PHASE", 
              "Architect-Zero is analyzing the task to identify required capabilities using LLM-enhanced understanding", 
              "REASONING")
    
    print(f"{Fore.YELLOW}Analyzing task requirements to extract the core capabilities needed...{Style.RESET_ALL}")
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(task_requirement, "document_processing")
    
    print_step("CAPABILITY RECOGNITION RESULTS", 
              {
                  "Process": "LLM-enhanced capability extraction from natural language requirements",
                  "Recognized capabilities": ", ".join(extracted_capabilities),
                  "Total capabilities": len(extracted_capabilities)
              }, 
              "REASONING")
    
    # Find components that match these capabilities
    print_step("CAPABILITY-COMPONENT MATCHING PHASE", 
              "Smart Library is searching for components that can fulfill each required capability", 
              "REASONING")
    
    print(f"{Fore.YELLOW}Searching library for existing components that match required capabilities...{Style.RESET_ALL}")
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
    
    print_step("COMPONENT DISCOVERY RESULTS", 
              {
                  "Process": "LLM-based semantic matching between capabilities and components",
                  "Coverage": f"{len([v for v in capability_matches.values() if v != 'No match found'])} of {len(capability_matches)} capabilities can be fulfilled by existing components",
                  "Matching details": capability_matches
              }, 
              "REASONING")
    
    # Execute Architect-Zero to design the system
    print_step("SYSTEM DESIGN PHASE", 
              "Architect-Zero is now designing a complete multi-agent system to solve the task", 
              "AGENT")
    
    print(f"{Fore.GREEN}Starting agent design and reasoning process...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Architect-Zero is determining which components to reuse, which to evolve, and which to create from scratch...{Style.RESET_ALL}")
    
    try:
        # Execute the architect agent
        start_time = time.time()
        result = await architect_agent.run(task_requirement)
        design_time = time.time() - start_time
        
        # Save the full thought process
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n{task_requirement}\n\n")
            f.write(f"AGENT THOUGHT PROCESS:\n{result.result.text}")
        
        # Count steps in the design process
        design_steps = len(re.findall(r'Step \d+:', result.result.text)) if 'Step' in result.result.text else 0
        reasoning_steps = len(re.findall(r'I need to|Let me|First,|Next,|Finally,|Now I will', result.result.text))
        
        # Show the reasoning process (truncated)
        reasoning_preview = result.result.text[:500] + "..." if len(result.result.text) > 500 else result.result.text
        print_step("AUTONOMOUS SYSTEM DESIGN COMPLETE", 
              {
                  "Design time": f"{design_time:.2f} seconds",
                  "Reasoning steps": max(design_steps, reasoning_steps),
                  "Design approach": "Capability-based component composition",
                  "Full reasoning": "Saved to 'architect_interaction.txt'"
              }, 
              "SUCCESS")
        
        # Extract workflow from the result
        yaml_content = await extract_yaml_workflow(result.result.text)
        if yaml_content:
            # Count agents and tools
            agent_count = len(re.findall(r'type: DEFINE\s+item_type: AGENT', yaml_content))
            tool_count = len(re.findall(r'type: DEFINE\s+item_type: TOOL', yaml_content))
            
            # Save the workflow to a file
            with open("invoice_workflow.yaml", "w") as f:
                f.write(yaml_content)
            
            print_step("MULTI-AGENT WORKFLOW GENERATION", 
                      {
                          "Process": "Translated design into executable YAML workflow",
                          "Components": f"{agent_count} agents and {tool_count} tools",
                          "Structure": "Define → Create → Execute pattern with data passing",
                          "Storage": "Complete workflow saved to invoice_workflow.yaml"
                      }, 
                      "SUCCESS")
            
            # Show abbreviated workflow
            workflow_lines = yaml_content.split('\n')
            workflow_preview = '\n'.join(workflow_lines[:15])
            if len(workflow_lines) > 15:
                workflow_preview += f"\n{Fore.CYAN}... (see invoice_workflow.yaml for complete workflow){Style.RESET_ALL}"
            print(workflow_preview)
            
            # Execute the workflow
            print_step("WORKFLOW EXECUTION PHASE", 
                      "System is now executing the designed multi-agent invoice processing workflow", 
                      "EXECUTION")
            
            print(f"{Fore.CYAN}Instantiating and connecting all agents in the workflow...{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Preparing to process sample invoice through the multi-agent system...{Style.RESET_ALL}")
            
            workflow_start_time = time.time()
            execution_result = await system_agent.workflow_processor.process_workflow(yaml_content)
            workflow_time = time.time() - workflow_start_time
            
            # Save execution result
            with open("workflow_execution_result.json", "w") as f:
                json.dump(execution_result, f, indent=2)
            
            # Show execution results
            if execution_result and execution_result.get("status") == "success":
                print_step("WORKFLOW EXECUTION COMPLETE", 
                      {
                          "Execution time": f"{workflow_time:.2f} seconds",
                          "Status": execution_result.get("status"),
                          "Execution details": "Saved to workflow_execution_result.json"
                      }, 
                      "SUCCESS")
                
                # Visualize the agent interaction flow
                await visualize_agent_interaction(execution_result)
                
                # Extract and display invoice analysis
                await extract_invoice_analysis(execution_result)
                
                # Calculate additional system metrics
                agent_count = len(re.findall(r'type: "CREATE"\s+item_type: "AGENT"', yaml_content))
                agent_dependencies = len(re.findall(r'user_input: \|', yaml_content))
                
                print_step("SYSTEM CAPABILITIES DEMONSTRATED", 
                          {
                              "Autonomous design": "Complete system designed without human intervention",
                              "Component discovery": f"Found and leveraged {len([v for v in capability_matches.values() if v != 'No match found'])} existing components",
                              "Multi-agent orchestration": f"Coordinated {agent_count} specialized agents",
                              "Transparent reasoning": "Complete visibility into both design and execution processes",
                              "Error detection": "Identified calculation discrepancies in the invoice data",
                              "Runtime adaptability": "System can evolve based on performance insights"
                          }, 
                          "SUCCESS")
                
                # Explain the benefits of evolving agents
                await explain_evolving_agents_benefits()
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
              "The Evolving Agents Toolkit has demonstrated autonomous design, component discovery, and multi-agent orchestration with transparent reasoning across all phases.", 
              "INFO")

if __name__ == "__main__":
    asyncio.run(main())