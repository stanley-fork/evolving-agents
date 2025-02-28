# examples/run_medical_diagnosis.py

import asyncio
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    # Initialize components
    library = SmartLibrary("library.json")
    llm = LLMService(provider="openai")
    system = SystemAgent(library, llm)
    
    # Initialize the system
    await system.initialize_system("config/system_config.yaml")
    
    # Two options to run:
    
    # Option 1: Process existing YAML
    print("Processing existing workflow...")
    result = await system.process_yaml_workflow("examples/medical_diagnosis/medical_scenario.yaml")
    
    # Option 2: Generate and execute from natural language
    print("\nGenerating workflow from natural language...")
    nl_result = await system.process_request(
        request="Create an agent that can analyze a patient's symptoms, provide a preliminary diagnosis for lupus, and research the latest immunosuppressant treatments. Include medical disclaimers.",
        domain="medical",
        output_yaml_path="examples/generated_workflow.yaml"
    )
    
    # Print results
    print("\nYAML workflow execution results:")
    for i, step in enumerate(result["execution"]["steps"]):
        print(f"Step {i+1}: {step.get('message', 'No message')}")
        if "result" in step:
            print(f"Result: {step['result']}")
    
    print("\nNatural language request results:")
    print(f"Generated YAML: {nl_result['workflow_yaml']}")
    for i, step in enumerate(nl_result["execution"]["steps"]):
        print(f"Step {i+1}: {step.get('message', 'No message')}")
        if "result" in step:
            print(f"Result: {step['result']}")

if __name__ == "__main__":
    asyncio.run(main())