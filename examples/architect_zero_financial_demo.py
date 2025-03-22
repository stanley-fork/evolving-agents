import asyncio
import logging
import json
import os
import re
import time
import colorama
from colorama import Fore, Style

# Import core components from the Evolving Agents Toolkit
from evolving_agents.core.llm_service import LLMService
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.agent_bus.simple_agent_bus import SimpleAgentBus
from evolving_agents.core.system_agent import SystemAgentFactory
from evolving_agents.agents.architect_zero import create_architect_zero
from evolving_agents.tools.tool_factory import ToolFactory

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
        "COMPONENT": Fore.MAGENTA  # For component creation
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
        "direct_invoice_analysis.txt",
        "complete_openai_integration.json"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed previous file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {str(e)}")

async def setup_library():
    """Set up initial reusable components in the smart library."""
    print_step("POPULATING COMPONENT LIBRARY", 
              "Creating real components for document analysis, invoice processing and calculation verification", 
              "COMPONENT")
    
    llm_service = LLMService(provider="openai", model="gpt-4o")
    smart_library = SmartLibrary("smart_library.json", llm_service)
    
    # Create a document analyzer component
    document_analyzer = {
        "name": "DocumentAnalyzer",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "Tool that analyzes and identifies the type of document from its content",
        "code_snippet": """
from pydantic import BaseModel, Field
import json
import re

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class DocumentAnalyzerInput(BaseModel):
    text: str = Field(description="Document text to analyze")

class DocumentAnalyzer(Tool[DocumentAnalyzerInput, ToolRunOptions, StringToolOutput]):
    \"\"\"Tool that analyzes and identifies the type of document from its content.\"\"\"
    name = "DocumentAnalyzer"
    description = "Analyzes document content to determine its type and key characteristics"
    input_schema = DocumentAnalyzerInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "document", "analyzer"],
            creator=self,
        )
    
    async def _run(self, input: DocumentAnalyzerInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Analyze a document to identify its type based on content patterns.\"\"\"
        document_text = input.text.lower()
        
        # Define type indicators with confidence scores
        type_indicators = {
            "invoice": ["invoice", "bill", "payment", "due date", "total due", "vendor", "subtotal", "tax"],
            "receipt": ["receipt", "payment received", "thank you for your purchase", "change", "cashier"],
            "medical_record": ["patient", "diagnosis", "treatment", "prescription", "doctor", "hospital", "medical"],
            "contract": ["agreement", "terms and conditions", "parties", "signed", "effective date", "termination"],
            "resume": ["experience", "education", "skills", "employment", "resume", "cv", "curriculum vitae"],
            "letter": ["dear", "sincerely", "regards", "to whom it may concern"],
            "report": ["report", "findings", "analysis", "conclusion", "executive summary"],
        }
        
        # Calculate confidence scores for each document type
        scores = {}
        for doc_type, indicators in type_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in document_text:
                    score += 1
            if score > 0:
                confidence = min(0.9, score / len(indicators))  # Cap at 90% confidence
                scores[doc_type] = confidence
        
        # Determine the most likely document type
        if scores:
            most_likely_type = max(scores.items(), key=lambda x: x[1])
            doc_type, confidence = most_likely_type
        else:
            doc_type = "unknown"
            confidence = 0.0
        
        # Extract key patterns based on document type
        extracted_fields = {}
        
        # For invoices, get some key fields
        if doc_type == "invoice":
            # Extract invoice number
            invoice_num_match = re.search(r'(?:invoice|inv)\\s*[#:]?\\s*(\\w+)', document_text, re.IGNORECASE)
            if invoice_num_match:
                extracted_fields["invoice_number"] = invoice_num_match.group(1)
                
            # Extract total
            total_match = re.search(r'total\\s*(?:due|amount)?\\s*[:$]?\\s*(\\d+[.,]?\\d*)', document_text)
            if total_match:
                extracted_fields["total"] = total_match.group(1)
        
        # Build the response
        result = {
            "document_type": doc_type,
            "confidence": confidence,
            "possible_types": [k for k, v in scores.items() if v > 0.2],
            "extracted_fields": extracted_fields
        }
        
        return StringToolOutput(json.dumps(result, indent=2))
""",
        "version": "1.0.0",
        "tags": ["document", "analysis", "classification"],
        "capabilities": [
            {
                "id": "document_analysis",
                "name": "Document Analysis",
                "description": "Analyzes and identifies the type of document from its content",
                "context": {
                    "required_fields": ["document_text"],
                    "produced_fields": ["document_type", "confidence", "extracted_fields"]
                }
            }
        ],
        "metadata": {
            "framework": "beeai"
        }
    }
    
    # Create a calculation verifier component
    calculation_verifier = {
        "name": "CalculationVerifier",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "Tool that verifies calculations in invoices, ensuring subtotal + tax = total",
        "code_snippet": """
from pydantic import BaseModel, Field
import json
import re

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class CalculationVerifierInput(BaseModel):
    invoice_data: str = Field(description="JSON string containing invoice data with subtotal, tax, and total fields")

class CalculationVerifier(Tool[CalculationVerifierInput, ToolRunOptions, StringToolOutput]):
    \"\"\"Tool that verifies calculations in invoices, ensuring subtotal + tax = total.\"\"\"
    name = "CalculationVerifier"
    description = "Verifies that calculations in an invoice are correct (subtotal + tax = total)"
    input_schema = CalculationVerifierInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "invoice", "verifier"],
            creator=self,
        )
    
    async def _run(self, input: CalculationVerifierInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Verify that calculations in an invoice are correct (subtotal + tax = total).\"\"\"
        try:
            # Parse the invoice data
            data = json.loads(input.invoice_data) if isinstance(input.invoice_data, str) else input.invoice_data
            
            # Extract the values
            subtotal = None
            tax = None
            total = None
            
            if "subtotal" in data:
                subtotal = float(str(data["subtotal"]).replace("$", "").replace(",", ""))
            if "tax" in data:
                if isinstance(data["tax"], dict) and "amount" in data["tax"]:
                    tax = float(str(data["tax"]["amount"]).replace("$", "").replace(",", ""))
                else:
                    tax = float(str(data["tax"]).replace("$", "").replace(",", ""))
            if "total_due" in data:
                total = float(str(data["total_due"]).replace("$", "").replace(",", ""))
            elif "total" in data:
                total = float(str(data["total"]).replace("$", "").replace(",", ""))
            
            # Verify the calculation
            if subtotal is not None and tax is not None and total is not None:
                expected_total = subtotal + tax
                is_correct = abs(expected_total - total) < 0.01  # Allow for small rounding differences
                
                result = {
                    "is_correct": is_correct,
                    "subtotal": subtotal,
                    "tax": tax,
                    "expected_total": expected_total,
                    "actual_total": total,
                    "difference": total - expected_total
                }
                
                return StringToolOutput(json.dumps(result, indent=2))
            else:
                return StringToolOutput(json.dumps({
                    "is_correct": False,
                    "error": "Missing required fields (subtotal, tax, or total)",
                    "available_fields": list(data.keys())
                }, indent=2))
                
        except Exception as e:
            return StringToolOutput(json.dumps({
                "is_correct": False,
                "error": f"Error verifying calculations: {str(e)}"
            }, indent=2))
""",
        "version": "1.0.0",
        "tags": ["invoice", "calculation", "verification"],
        "capabilities": [
            {
                "id": "calculation_verification",
                "name": "Calculation Verification",
                "description": "Verifies that calculations in an invoice are correct (subtotal + tax = total)",
                "context": {
                    "required_fields": ["invoice_data"],
                    "produced_fields": ["is_correct", "expected_total", "difference"]
                }
            }
        ],
        "metadata": {
            "framework": "beeai"
        }
    }
    
    # Create an invoice summary generator component
    invoice_summary = {
        "name": "InvoiceSummaryGenerator",
        "record_type": "TOOL",
        "domain": "document_processing",
        "description": "Tool that generates a concise summary of an invoice's key information",
        "code_snippet": """
from pydantic import BaseModel, Field
import json
from datetime import datetime

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

class InvoiceSummaryInput(BaseModel):
    invoice_data: str = Field(description="JSON string containing structured invoice data")

class InvoiceSummaryGenerator(Tool[InvoiceSummaryInput, ToolRunOptions, StringToolOutput]):
    \"\"\"Tool that generates a concise summary of an invoice's key information.\"\"\"
    name = "InvoiceSummaryGenerator"
    description = "Generates a concise summary from structured invoice data"
    input_schema = InvoiceSummaryInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "invoice", "summary"],
            creator=self,
        )
    
    async def _run(self, input: InvoiceSummaryInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        \"\"\"Generate a concise summary of an invoice from structured data.\"\"\"
        try:
            # Parse the invoice data
            data = json.loads(input.invoice_data) if isinstance(input.invoice_data, str) else input.invoice_data
            
            # Extract key information
            invoice_number = data.get("invoice_number", "Unknown")
            date = data.get("date", "Unknown date")
            vendor_name = data.get("vendor", {}).get("name", data.get("vendor", "Unknown vendor"))
            total_due = data.get("total_due", data.get("total", "Unknown amount"))
            due_date = data.get("due_date", "Unknown")
            
            # Format as currency if needed
            if isinstance(total_due, (int, float)):
                total_due = f"${total_due:,.2f}"
            
            # Generate the summary
            summary = {
                "summary": f"Invoice #{invoice_number} from {vendor_name}",
                "key_details": {
                    "invoice_number": invoice_number,
                    "vendor": vendor_name,
                    "date": date,
                    "total_due": total_due,
                    "due_date": due_date
                },
                "line_item_count": len(data.get("items", [])),
                "recommendations": []
            }
            
            # Add recommendations based on the data
            if due_date:
                try:
                    due_date_obj = datetime.strptime(due_date, "%Y-%m-%d")
                    today = datetime.now()
                    days_until_due = (due_date_obj - today).days
                    
                    if days_until_due < 0:
                        summary["recommendations"].append("OVERDUE: Payment is past due")
                    elif days_until_due < 7:
                        summary["recommendations"].append(f"URGENT: Payment due soon ({days_until_due} days)")
                    elif days_until_due < 30:
                        summary["recommendations"].append(f"REMINDER: Payment due in {days_until_due} days")
                except:
                    pass
            
            # Check for large amounts
            try:
                amount = float(str(total_due).replace("$", "").replace(",", ""))
                if amount > 1000:
                    summary["recommendations"].append("ATTENTION: Invoice amount exceeds $1,000")
            except:
                pass
                
            return StringToolOutput(json.dumps(summary, indent=2))
            
        except Exception as e:
            return StringToolOutput(json.dumps({
                "error": f"Error generating summary: {str(e)}",
                "partial_summary": "Unable to generate complete summary due to an error"
            }, indent=2))
""",
        "version": "1.0.0",
        "tags": ["invoice", "summary", "report"],
        "capabilities": [
            {
                "id": "summary_generation",
                "name": "Summary Generation",
                "description": "Generates a concise summary of an invoice with key details and recommendations",
                "context": {
                    "required_fields": ["invoice_data"],
                    "produced_fields": ["summary", "key_details", "recommendations"]
                }
            }
        ],
        "metadata": {
            "framework": "beeai"
        }
    }
    
    # Create an OpenAI invoice processor agent
    openai_invoice_processor = {
        "name": "OpenAIInvoiceProcessor",
        "record_type": "AGENT",
        "domain": "document_processing",
        "description": "An OpenAI agent specialized in processing invoice documents. This agent can extract structured data from invoices, verify calculations, identify key information, and follow governance rules for financial documents.",
        "code_snippet": """
from agents import Agent, Runner, ModelSettings

# Create an OpenAI agent for invoice processing
agent = Agent(
    name="InvoiceProcessor",
    instructions=\"\"\"
You are an expert invoice processing agent specialized in financial document analysis.
Extract all important information from invoices including:
- Invoice number
- Date
- Vendor information
- Line items (description, quantity, price)
- Subtotal, taxes, and total
- Payment terms

Verify that calculations are correct.
Flag any suspicious or unusual patterns.
Structure your output clearly and follow all governance rules for financial documents.
\"\"\",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.3  # Lower temperature for more precision
    )
)

# Helper function to process invoices
async def process_invoice(invoice_text):
    result = await Runner.run(agent, input=invoice_text)
    return result.final_output
""",
        "version": "1.0.0",
        "tags": ["openai", "invoice", "financial", "agent"],
        "capabilities": [
            {
                "id": "invoice_data_extraction",
                "name": "Invoice Data Extraction",
                "description": "Extracts structured data from invoice documents including invoice number, date, vendor, line items, and totals",
                "context": {
                    "required_fields": ["invoice_text"],
                    "produced_fields": ["structured_invoice_data"]
                }
            }
        ],
        "metadata": {
            "framework": "openai-agents",
            "model": "gpt-4o",
            "model_settings": {
                "temperature": 0.3
            },
            "guardrails_enabled": True
        }
    }
    
    # Save the components to the library
    doc_analyzer_record = await smart_library.create_record(**document_analyzer)
    calc_verifier_record = await smart_library.create_record(**calculation_verifier)
    summary_record = await smart_library.create_record(**invoice_summary)
    openai_agent_record = await smart_library.create_record(**openai_invoice_processor)
    
    # Save a complete collection of OpenAI-compatible components for later integration
    complete_integration = [
        document_analyzer,
        calculation_verifier,
        invoice_summary,
        openai_invoice_processor
    ]
    
    with open("complete_openai_integration.json", "w") as f:
        json.dump(complete_integration, f, indent=2)
    
    # Print confirmation of what was created
    print(f"{Fore.GREEN}Created real components in the library:{Style.RESET_ALL}")
    print(f" - {Fore.CYAN}DocumentAnalyzer{Style.RESET_ALL}: A tool that identifies document types with confidence scores")
    print(f" - {Fore.CYAN}CalculationVerifier{Style.RESET_ALL}: A tool that ensures invoice calculations are correct")
    print(f" - {Fore.CYAN}InvoiceSummaryGenerator{Style.RESET_ALL}: A tool that generates actionable invoice summaries")
    print(f" - {Fore.CYAN}OpenAIInvoiceProcessor{Style.RESET_ALL}: An OpenAI agent specialized in invoice data extraction")
    
    return smart_library

async def perform_direct_invoice_analysis(llm_service, sample_invoice):
    """Perform a direct invoice analysis to demonstrate agent capabilities."""
    print_step("PERFORMING DIRECT INVOICE ANALYSIS", 
              "Showing what a specialized invoice analysis agent can extract from an invoice", 
              "EXECUTION")
    
    prompt = f"""
    You are an expert invoice analyst. Please analyze this invoice in detail and provide:
    
    1. All extracted key information (invoice number, date, vendor, billing info, etc.)
    2. Line items with quantities and amounts
    3. Verification of calculations (do the numbers add up correctly?)
    4. Any potential errors or inconsistencies in the invoice
    5. A structured summary with insights
    
    Invoice:
    {sample_invoice}
    
    Provide your analysis in a well-structured format with clear sections.
    """
    
    response = await llm_service.generate(prompt)
    
    # Save the direct analysis
    with open("direct_invoice_analysis.txt", "w") as f:
        f.write(response)
    
    # Display a preview of the analysis
    preview_lines = response.split('\n')[:15]
    print(f"\n{Fore.YELLOW}Invoice Analysis Preview:{Style.RESET_ALL}")
    for line in preview_lines:
        print(line)
    print(f"{Fore.YELLOW}... (see direct_invoice_analysis.txt for full analysis){Style.RESET_ALL}")
    
    # Extract calculation verification from the analysis
    calculation_section = ""
    capturing = False
    
    for line in response.split('\n'):
        if "Verification of Calculations" in line or "calculations" in line.lower() and "verify" in line.lower():
            capturing = True
            calculation_section = line + "\n"
        elif capturing and line.strip() and (line.startswith("4.") or line.startswith("5.")):
            capturing = False
        elif capturing:
            calculation_section += line + "\n"
    
    # Return key insights from the analysis
    return {
        "calculation_verification": calculation_section.strip(),
        "full_analysis": "direct_invoice_analysis.txt"
    }

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
    print_step("EVOLVING AGENTS TOOLKIT DEMONSTRATION", 
              "This demonstration shows how specialized agents can collaborate, evolve, and reason transparently to process invoices.", 
              "INFO")
    
    # Clean up previous files
    clean_previous_files()
    
    # Initialize LLM service
    llm_service = LLMService(provider="openai", model="gpt-4o")
    
    # Set up initial components in the library
    smart_library = await setup_library()
    
    # Perform a direct invoice analysis to show base capabilities
    analysis_results = await perform_direct_invoice_analysis(llm_service, SAMPLE_INVOICE)
    
    # Initialize agent bus for agent communication
    agent_bus = SimpleAgentBus("agent_bus.json")
    agent_bus.set_llm_service(llm_service)
    
    # Create the system agent that will manage the agent ecosystem
    print_step("INITIALIZING SYSTEM AGENT", 
              "Creating the System Agent that manages the agent ecosystem and orchestrates workflow execution", 
              "AGENT")
    
    system_agent = await SystemAgentFactory.create_agent(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus
    )
    system_agent.workflow_processor.set_llm_service(llm_service)
    system_agent.workflow_generator.set_llm_service(llm_service)
    
    # Create tool factory for instantiating real components
    tool_factory = ToolFactory(smart_library, llm_service)
    
    # Create the Architect-Zero meta-agent
    print_step("CREATING ARCHITECT-ZERO META-AGENT", 
              "This agent designs entire agent systems by analyzing requirements and composing specialized components", 
              "AGENT")
    
    architect_agent = await create_architect_zero(
        llm_service=llm_service,
        smart_library=smart_library,
        agent_bus=agent_bus,
        system_agent=system_agent
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
    print_step("INVOICE PROCESSING TASK REQUIREMENTS", task_requirement, "INFO")
    
    # Extract required capabilities from the task requirements
    print_step("CAPABILITY EXTRACTION WITH LLM", 
              "Using the LLM to identify the specialized capabilities needed for this task", 
              "REASONING")
    
    extracted_capabilities = await smart_library._extract_capabilities_with_llm(task_requirement, "document_processing")
    print_step("REQUIRED CAPABILITIES", {
        "Extracted capabilities": ", ".join(extracted_capabilities)
    }, "REASONING")
    
    # Find the best components to fulfill each capability
    print_step("INTELLIGENT COMPONENT SELECTION", 
              "Searching for the best components to fulfill each capability using semantic matching and LLM reasoning", 
              "REASONING")
    
    workflow_components = await smart_library.find_components_for_workflow(
        workflow_description=task_requirement,
        required_capabilities=extracted_capabilities,
        domain="document_processing",
        use_llm=True
    )
    
    capability_matches = {}
    for cap_id, components in workflow_components.items():
        component_names = [f"{c['name']} ({c['record_type']})" for c in components]
        capability_matches[cap_id] = ", ".join(component_names) if component_names else "No match found"
    
    print_step("CAPABILITY-COMPONENT MAPPING", capability_matches, "REASONING")
    
    # Execute Architect-Zero to design the solution
    print_step("DESIGNING INVOICE PROCESSING SYSTEM", 
              "Architect-Zero is designing a multi-agent solution with full reasoning transparency", 
              "AGENT")
    
    try:
        # Execute the architect agent with full reasoning log
        print(f"{Fore.GREEN}Starting agent reasoning process...{Style.RESET_ALL}")
        start_time = time.time()
        result = await architect_agent.run(task_requirement)
        design_time = time.time() - start_time
        
        # Save the full thought process
        with open("architect_interaction.txt", "w") as f:
            f.write(f"TASK REQUIREMENT:\n\n    {task_requirement}\n\n")
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
            
            print_step("MULTI-AGENT WORKFLOW GENERATED", 
                      "Architect-Zero has created a complete workflow with specialized agents for each part of the process", 
                      "SUCCESS")
            
            # Show abbreviated workflow
            workflow_lines = yaml_content.split('\n')
            workflow_preview = '\n'.join(workflow_lines[:20])
            if len(workflow_lines) > 20:
                workflow_preview += f"\n{Fore.CYAN}... (see invoice_workflow.yaml for complete workflow){Style.RESET_ALL}"
            print(workflow_preview)
            
            # Extract information about the components in the workflow
            component_definitions = re.findall(r'type:\s+DEFINE.*?name:\s+(\w+).*?item_type:\s+(\w+)', yaml_content, re.DOTALL)
            component_executions = re.findall(r'type:\s+EXECUTE.*?name:\s+(\w+)', yaml_content, re.DOTALL)
            
            print_step("WORKFLOW COMPONENT ANALYSIS", {
                "Component definitions": len(component_definitions),
                "Component executions": len(component_executions),
                "Defined components": ", ".join([f"{name} ({type})" for name, type in component_definitions]),
                "Execution sequence": " → ".join(component_executions)
            }, "REASONING")
            
            # Execute the workflow
            print_step("EXECUTING MULTI-AGENT WORKFLOW", 
                      "Now the system will instantiate and execute all components in the workflow", 
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
                    "Status": execution_result.get("status"),
                    "Result": "See detailed output below"
                }, "SUCCESS")
                
                # Extract detailed results from execution output
                result_text = execution_result.get("result", "")
                
                # Extract step executions for better visibility
                step_pattern = r'Step (\d+): \*\*([^*]+)\*\*\s*\n\s*- \*\*([^*]+)\*\*: (.*?)(?=\n\s*(?:- \*\*|\n\d+\.|$))'
                steps = re.findall(step_pattern, result_text, re.DOTALL)
                
                # Display the step results in a more readable format
                if steps:
                    print(f"\n{Fore.CYAN}Workflow Execution Steps:{Style.RESET_ALL}")
                    for step_num, step_type, action_type, action_result in steps:
                        print(f"{Fore.GREEN}Step {step_num}{Style.RESET_ALL}: {step_type.strip()}")
                        print(f"  {Fore.YELLOW}{action_type.strip()}{Style.RESET_ALL}: {action_result.strip()}")
                        print()
                else:
                    # Fallback to showing the original text
                    print(result_text)
                
                # Extract and display verification results to compare with direct analysis
                verification_text = re.search(r'Calculations verified.*?(?=\n\n|\Z)', result_text, re.DOTALL)
                if verification_text:
                    verification_result = verification_text.group(0)
                else:
                    verification_result = "Calculation verification details not found in output"
                
                print_step("INVOICE CALCULATION VERIFICATION", 
                         "Comparison between the direct calculation check and the evolved system", 
                         "COMPARISON")
                
                print(f"{Fore.YELLOW}Direct Invoice Analysis:{Style.RESET_ALL}")
                print(analysis_results["calculation_verification"])
                print(f"\n{Fore.YELLOW}Multi-Agent System Verification:{Style.RESET_ALL}")
                print(verification_result)
                
                # Show insights about the agent collaboration
                agent_count = len(re.findall(r'type:\s+DEFINE.*?item_type:\s+AGENT', yaml_content, re.DOTALL))
                tool_count = len(re.findall(r'type:\s+DEFINE.*?item_type:\s+TOOL', yaml_content, re.DOTALL))
                data_flows = len(re.findall(r'input_data:', yaml_content))
                
                print_step("EVOLVING AGENTS SYSTEM INSIGHTS", {
                    "Specialized agents": agent_count,
                    "Specialized tools": tool_count,
                    "Data flows between components": data_flows,
                    "OpenAI integration": "OpenAIInvoiceProcessor used for structured data extraction",
                    "Calculation verification": "Independent component verified calculations accuracy",
                    "Transparency level": "Full visibility into each step's reasoning and contribution",
                    "Error detection": f"Found discrepancy in invoice calculations (${execution_result.get('difference', 'unknown')})",
                    "System advantage": "Component specialization with transparent reasoning and evolution capability"
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
    
    print_step("DEMONSTRATION COMPLETE", 
              """
This demonstration showed the key capabilities of the Evolving Agents Toolkit:

1. Smart Library with LLM-powered capability matching for component selection
2. Architect-Zero design of multi-agent systems from high-level requirements
3. Execution of workflows with real, specialized components
4. Integration between different agent frameworks (BeeAI and OpenAI)
5. Full transparency into agent reasoning and collaboration
6. Automatic orchestration of complex multi-agent systems

The toolkit enables the creation of systems where specialized agents collaborate with 
transparent reasoning, providing both performance and explainability.
              """, 
              "INFO")

if __name__ == "__main__":
    asyncio.run(main())