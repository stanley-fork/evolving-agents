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
        
        # Initialize the agent and tool factories with the provider registry
        agent_factory = AgentFactory(library, llm_service, provider_registry)
        tool_factory = ToolFactory(library, llm_service)
        
        # Initialize system agent with proper factories
        system_agent = SystemAgent(
            smart_library=library, 
            llm_service=llm_service,
            agent_factory=agent_factory,
            tool_factory=tool_factory,
            provider_registry=provider_registry
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

  # Create the coordinator agent with the tools
  - type: "CREATE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    config:
      memory_type: "token"

  # Execute with an invoice document
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    user_input: "Process this document: {invoice}"
    execution_config:
      max_iterations: 15
"""
        
        # Replace placeholders with sample documents
        workflow_yaml = workflow_yaml.replace("{invoice}", SAMPLE_INVOICE.replace("\n", "\\n"))
        
        # Process the workflow
        print("\nExecuting workflow for invoice processing...")
        invoice_results = await workflow_processor.process_workflow(workflow_yaml)
        
        # Print execution results
        print("\nWorkflow Execution Results:")
        print("="*80)
        
        for i, step in enumerate(invoice_results.get("steps", [])):
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
                
                # Try to pretty print JSON if possible
                try:
                    result_json = json.loads(result_text)
                    print(f"    {json.dumps(result_json, indent=2)}")
                except:
                    print("    " + result_text.replace("\n", "\n    "))
            
            print("-" * 40)
        
        # Now try with a medical record
        medical_workflow_yaml = workflow_yaml.replace("{invoice}", SAMPLE_MEDICAL_RECORD.replace("\n", "\\n"))
        
        print("\nExecuting workflow for medical record processing...")
        medical_results = await workflow_processor.process_workflow(medical_workflow_yaml)
        
        # Print execution results
        print("\nMedical Record Workflow Results:")
        print("="*80)
        
        # Show only the final result for brevity
        last_step = medical_results.get("steps", [])[-1]
        if "result" in last_step and last_step.get("status") == "success":
            print("Final Result:")
            result_text = last_step["result"]
            
            # Try to pretty print JSON if possible
            try:
                result_json = json.loads(result_text)
                print(f"  {json.dumps(result_json, indent=2)}")
            except:
                print("  " + result_text.replace("\n", "\n  "))
        else:
            print(f"Error: {last_step.get('message', 'No result')}")
        
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
      docstring_update: "Improved with enhanced invoice analysis capabilities including line item detection"
    description: "Enhanced specialist that provides more detailed invoice analysis with line item extraction"

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
"""
        
        # Replace placeholder with invoice
        evolution_workflow = evolution_workflow.replace("{invoice}", SAMPLE_INVOICE.replace("\n", "\\n"))
        
        # Process the evolution workflow
        print("\nExecuting evolution workflow to create enhanced invoice specialist...")
        evolution_results = await workflow_processor.process_workflow(evolution_workflow)
        
        # Print evolution results
        print("\nEvolution Results:")
        print("="*80)
        
        # Show only the final execution result for brevity
        last_step = evolution_results.get("steps", [])[-1]
        if "result" in last_step and last_step.get("status") == "success":
            print("Enhanced Invoice Analysis Result:")
            result_text = last_step["result"]
            
            # Try to pretty print JSON if possible
            try:
                result_json = json.loads(result_text)
                print(f"  {json.dumps(result_json, indent=2)}")
            except:
                print("  " + result_text.replace("\n", "\n  "))
        else:
            print(f"Error: {last_step.get('message', 'No result')}")

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