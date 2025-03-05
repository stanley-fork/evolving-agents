# examples/simplified_agent_communication.py

import asyncio
import logging
import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider
from evolving_agents.tools.tool_factory import ToolFactory
from evolving_agents.agents.agent_factory import AgentFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample document data
SAMPLE_INVOICE = """
INVOICE #12345
Date: 2023-05-15
Vendor: TechSupplies Inc.
Address: 123 Business Ave, Commerce City, CA 90210

Items:
1. Laptop Computer - $1,200.00 (2 units)
2. Wireless Mouse - $25.00 (5 units)
3. External Hard Drive - $85.00 (3 units)

Subtotal: $1,680.00
Tax (8.5%): $142.80
Total Due: $1,822.80

Payment Terms: Net 30
Due Date: 2023-06-14
"""

SAMPLE_MEDICAL_RECORD = """
PATIENT MEDICAL RECORD
Patient ID: P789456
Name: John Smith
DOB: 1975-03-12
Visit Date: 2023-05-10

Chief Complaint: Patient presents with persistent cough for 2 weeks, mild fever, and fatigue.

Vitals:
- Temperature: 100.2°F
- Blood Pressure: 128/82
- Heart Rate: 88 bpm
- Respiratory Rate: 18/min
- Oxygen Saturation: 97%

Assessment: Acute bronchitis
Plan: Prescribed antibiotics (Azithromycin 500mg) for 5 days, recommended rest and increased fluid intake.
Follow-up in 1 week if symptoms persist.
"""

# Fix for DocumentAnalyzer tool
FIXED_DOCUMENT_ANALYZER = '''
# Tool to analyze documents and identify their type
import json
import re

def analyze_document(input):
    """
    Analyzes a document to identify its type and key characteristics.
    
    Args:
        input: Document text to analyze
        
    Returns:
        Document analysis including type, confidence, and keywords
    """
    text = input.lower()
    result = {
        "document_type": "unknown",
        "confidence": 0.5,
        "keywords": []
    }
    
    # Extract keywords (words that appear frequently or seem important)
    words = text.split()
    word_counts = {}
    
    for word in words:
        # Clean the word - Fixed version with properly escaped quotes
        clean_word = word.strip(".,;:()[]{}\"'")  # Removed the problematic sequence
        if len(clean_word) > 3:  # Only count words with at least 4 characters
            word_counts[clean_word] = word_counts.get(clean_word, 0) + 1
    
    # Get the top 5 most frequent words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    result["keywords"] = [word for word, count in sorted_words[:5]]
    
    # Determine document type based on content
    if "invoice" in text:
        if "total" in text and ("payment" in text or "due" in text):
            result["document_type"] = "invoice"
            result["confidence"] = 0.9
            
            # Check for invoice amount
            money_pattern = r'\\$(\d+,?\d*\.\d{2})'
            amounts = re.findall(money_pattern, input)
            if amounts:
                result["has_monetary_values"] = True
                try:
                    result["highest_amount"] = max([float(amt.replace(",", "")) for amt in amounts])
                except:
                    pass
    
    elif "patient" in text:
        if "medical" in text or "assessment" in text or "diagnosis" in text:
            result["document_type"] = "medical"
            result["confidence"] = 0.92
            
            # Check for medical keywords
            medical_keywords = ["prescribed", "symptoms", "treatment", "follow-up", "medication"]
            for keyword in medical_keywords:
                if keyword in text:
                    result["keywords"].append(keyword)
    
    elif "contract" in text or "agreement" in text:
        result["document_type"] = "contract"
        result["confidence"] = 0.85
    
    elif "report" in text:
        result["document_type"] = "report"
        result["confidence"] = 0.7
    
    # Clean up keywords to remove duplicates and sort
    result["keywords"] = list(set(result["keywords"]))
    
    return result

# Process the input text
result = analyze_document(input)
'''

# Fix for SystemAgent.execute_item method
async def fixed_execute_item(self, name: str, input_text: str):
    """
    Execute an active item (agent or tool).
    
    Args:
        name: Name of the item to execute
        input_text: Input text for the item
        
    Returns:
        Execution result
    """
    if name not in self.active_items:
        return {
            "status": "error",
            "message": f"Item '{name}' not found in active items"
        }
    
    item = self.active_items[name]
    record = item["record"]
    instance = item["instance"]
    
    logger.info(f"Executing {record['record_type']} '{name}' with input: {input_text[:50]}...")
    
    try:
        # Execute based on record type
        if record["record_type"] == "AGENT":
            result = await self.agent_factory.execute_agent(
                name,  # Use name here, not instance
                input_text
            )
        else:  # TOOL
            result = await self.tool_factory.execute_tool(
                instance,
                input_text
            )
        
        # Update usage metrics
        await self.library.update_usage_metrics(record["id"], True)
        
        return {
            "status": "success",
            "item_name": name,
            "item_type": record["record_type"],
            "result": result,
            "message": f"Executed {record['record_type']} '{name}'"
        }
    except Exception as e:
        logger.error(f"Error executing {record['record_type']} '{name}': {str(e)}")
        
        # Update usage metrics as failure
        await self.library.update_usage_metrics(record["id"], False)
        
        return {
            "status": "error",
            "message": f"Error executing {record['record_type']} '{name}': {str(e)}"
        }

async def main():
    try:
        # Initialize components
        library_path = "simplified_agent_library.json"
        
        # Check if library exists, and if not, set up the library first
        if not os.path.exists(library_path):
            print("Library not found. Please run setup_simplified_agent_library.py first.")
            print("Alternatively, we can set up the library now. Would you like to proceed? (y/n)")
            response = input().lower()
            if response == 'y':
                # Import and run the setup function
                from setup_simplified_agent_library import main as setup_main
                await setup_main()
            else:
                print("Exiting. Please run setup_simplified_agent_library.py before this example.")
                return
        
        # Initialize the Smart Library and components
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        # Initialize the provider registry and add the BeeAI provider
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        
        # Fix DocumentAnalyzer tool
        print("\nApplying fixes to library components...")
        
        # Find and fix DocumentAnalyzer tool
        analyzer_record = None
        for record in library.records:
            if record["name"] == "DocumentAnalyzer" and record["record_type"] == "TOOL":
                analyzer_record = record
                break
        
        if analyzer_record:
            # Update the code snippet
            analyzer_record["code_snippet"] = FIXED_DOCUMENT_ANALYZER
            
            # Save it back to the library
            await library.save_record(analyzer_record)
            print(f"✓ Updated DocumentAnalyzer tool (ID: {analyzer_record['id']})")
        
        # Apply patch to SystemAgent.execute_item method
        SystemAgent.execute_item = fixed_execute_item
        print("✓ SystemAgent.execute_item method has been patched")
        
        # Initialize system agent with the provider registry
        system_agent = SystemAgent(
            smart_library=library, 
            llm_service=llm_service
        )
        
        # Initialize workflow processor
        workflow_processor = WorkflowProcessor(system_agent)
        
        print("\n" + "="*80)
        print("PART 1: DEMONSTRATING SYSTEM AGENT'S DECISION LOGIC")
        print("="*80)
        
        # First, demonstrate the system agent's decision logic
        print("\nAsking System Agent for an invoice analysis agent...")
        
        invoice_agent_result = await system_agent.decide_and_act(
            request="I need an agent that can analyze invoices and extract the total amount",
            domain="document_processing",
            record_type="AGENT"
        )
        
        print(f"System Agent Decision: {invoice_agent_result['action']}")
        print(f"Agent: {invoice_agent_result['record']['name']}")
        
        if 'similarity' in invoice_agent_result:
            print(f"Similarity Score: {invoice_agent_result['similarity']:.4f}")
        
        print(f"Message: {invoice_agent_result['message']}")
        
        # Now, ask for a medical record analyzer
        print("\nAsking System Agent for a medical record analysis agent...")
        
        medical_agent_result = await system_agent.decide_and_act(
            request="I need an agent that can analyze medical records and extract patient information",
            domain="document_processing",
            record_type="AGENT"
        )
        
        print(f"System Agent Decision: {medical_agent_result['action']}")
        print(f"Agent: {medical_agent_result['record']['name']}")
        
        if 'similarity' in medical_agent_result:
            print(f"Similarity Score: {medical_agent_result['similarity']:.4f}")
        
        print(f"Message: {medical_agent_result['message']}")
        
        # Now execute the invoice agent
        print("\nExecuting the invoice analysis agent...")
        
        invoice_execution = await system_agent.execute_item(
            invoice_agent_result['record']['name'],
            SAMPLE_INVOICE
        )
        
        if invoice_execution["status"] == "success":
            print(f"\nInvoice Analysis Result:")
            print(invoice_execution["result"])
        else:
            print(f"Error: {invoice_execution['message']}")
        
        # Now execute the medical agent
        print("\nExecuting the medical record analysis agent...")
        
        medical_execution = await system_agent.execute_item(
            medical_agent_result['record']['name'],
            SAMPLE_MEDICAL_RECORD
        )
        
        if medical_execution["status"] == "success":
            print(f"\nMedical Record Analysis Result:")
            print(medical_execution["result"])
        else:
            print(f"Error: {medical_execution['message']}")
            
        
        print("\n" + "="*80)
        print("PART 2: DEMONSTRATING AGENT-TO-AGENT COMMUNICATION WITH WORKFLOWS")
        print("="*80)
        
        # Define our workflow that uses the pre-stored agents and tools
        workflow_yaml = """
scenario_name: "Document Processing with Agent Communication"
domain: "document_processing"
description: "Process documents by delegating specialized tasks to expert agents"

steps:
  # Create the tools from the library
  - type: "CREATE"
    item_type: "TOOL"
    name: "DocumentAnalyzer"

  - type: "CREATE"
    item_type: "TOOL"
    name: "AgentCommunicator"

  # Create the agents from the library
  - type: "CREATE"
    item_type: "AGENT"
    name: "SpecialistAgent"
    config:
      memory_type: "token"

  - type: "CREATE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    config:
      memory_type: "token"

  # Test with an invoice document
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    user_input: "Process this document: {invoice}"
    execution_config:
      max_iterations: 15
      enable_observability: true
"""
        
        # Replace placeholders with sample documents
        workflow_yaml = workflow_yaml.replace("{invoice}", SAMPLE_INVOICE.replace("\n", "\\n"))
        
        print("\nWorkflow to Execute:")
        print(workflow_yaml)
        
        # Process the workflow
        print("\nExecuting workflow...")
        results = await workflow_processor.process_workflow(workflow_yaml)
        
        # Print execution results
        print("\nWorkflow Execution Results:")
        print("="*80)
        
        for i, step in enumerate(results.get("steps", [])):
            print(f"Step {i+1}: {step.get('message', 'No message')}")
            
            # Print execution details if available
            if "execution_details" in step and step["execution_details"]:
                details = step["execution_details"]
                print(f"  Framework: {details.get('framework', 'N/A')}")
                print(f"  Provider: {details.get('provider', 'N/A')}")
            
            # Print result if available (for EXECUTE steps)
            if "result" in step and step.get("status") == "success":
                print("\n  Result:")
                result_text = step["result"]
                
                # Try to parse JSON result
                try:
                    result_json = json.loads(result_text)
                    print(f"    {json.dumps(result_json, indent=2)}")
                except:
                    # Handle as text - Fix for backslash issue in f-strings
                    print("    " + result_text.replace("\n", "\n    "))
            
            print("-" * 40)
        
        print("\n" + "="*80)
        print("PART 3: DEMONSTRATING AGENT EVOLUTION")
        print("="*80)
        
        # Define an evolution workflow
        evolution_workflow = """
scenario_name: "Enhanced Invoice Processing"
domain: "document_processing"
description: "Evolve the specialist agent to provide better invoice analysis"

steps:
  # Define an evolved version of the specialist agent
  - type: "DEFINE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    from_existing_snippet: "SpecialistAgent"
    evolve_changes:
      docstring_update: "Improved with enhanced invoice analysis capabilities"
    description: "Enhanced specialist that provides more detailed invoice analysis"

  # Create the evolved agent
  - type: "CREATE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    config:
      memory_type: "token"

  # Test the evolved agent with an invoice
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    user_input: "{invoice}"
    execution_config:
      max_iterations: 15
      enable_observability: true
"""
        
        # Replace placeholder with invoice
        evolution_workflow = evolution_workflow.replace("{invoice}", SAMPLE_INVOICE.replace("\n", "\\n"))
        
        # Process the evolution workflow
        print("\nExecuting evolution workflow...")
        evolution_results = await workflow_processor.process_workflow(evolution_workflow)
        
        # Print evolution results
        print("\nEvolution Results:")
        print("="*80)
        
        for i, step in enumerate(evolution_results.get("steps", [])):
            print(f"Step {i+1}: {step.get('message', 'No message')}")
            
            # Print result for the execution step
            if "result" in step and i == 2:  # The execution step
                print("\n  Enhanced Analysis Result:")
                result_text = step["result"]
                
                # Try to parse JSON result
                try:
                    result_json = json.loads(result_text)
                    print(f"    {json.dumps(result_json, indent=2)}")
                except:
                    # Handle as text
                    print("    " + result_text.replace("\n", "\n    "))
            
            print("-" * 40)

        print("\n" + "="*80)
        print("PART 4: DEMONSTRATING SEMANTIC SEARCH WITH OPENAI EMBEDDINGS")
        print("="*80)
        
        print("\nSearching for document processing agents...")
        search_results = await library.semantic_search(
            query="agent that can process and understand documents",
            record_type="AGENT",
            threshold=0.3
        )
        
        print("\nSemantic Search Results:")
        for i, (record, score) in enumerate(search_results):
            print(f"Match {i+1}: {record['name']} (Score: {score:.4f})")
            print(f"  Description: {record['description']}")
            print(f"  Type: {record['record_type']}")
            print("-" * 40)
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())