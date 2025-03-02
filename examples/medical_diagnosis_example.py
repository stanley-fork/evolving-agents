# examples/medical_diagnosis_example.py

import asyncio
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.workflow.workflow_generator import WorkflowGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Hardcoded tool implementation for reliability
SYMPTOM_PARSER_CODE = """
# MEDICAL_DISCLAIMER: This tool is for informational purposes only and not a substitute for professional medical advice.

def parse_symptoms(input_text):
    \"\"\"
    Parse symptoms from free text into structured data.
    
    Args:
        input_text: Patient description of symptoms
        
    Returns:
        Dictionary of structured symptom data
    \"\"\"
    # Simple parsing logic
    text = input_text.lower()
    symptoms = []
    
    if "joint pain" in text:
        symptoms.append({"name": "joint pain", "severity": "unknown"})
    if "fatigue" in text:
        symptoms.append({"name": "fatigue", "severity": "unknown"})
    if "rash" in text:
        symptoms.append({"name": "skin rash", "severity": "unknown"})
    if "butterfly" in text:
        symptoms.append({"name": "butterfly rash", "severity": "unknown", "location": "face"})
    
    # Format the output
    output = {
        "symptoms": symptoms,
        "disclaimer": "This is an automated parsing of symptoms. Medical professionals should verify."
    }
    
    # Determine if symptoms match lupus pattern
    lupus_indicators = ["joint pain", "fatigue", "butterfly rash"]
    matched_indicators = [s["name"] for s in symptoms if s["name"] in lupus_indicators or 
                         (s["name"] == "skin rash" and "butterfly" in text)]
    
    if len(matched_indicators) >= 2:
        output["possible_conditions"] = ["Lupus (SLE)"]
        output["recommendation"] = "Consult with a rheumatologist for proper evaluation."
    
    return output

# Call the function with the input and store the result
result = parse_symptoms(input)
"""

async def main():
    try:
        # Initialize components with GPT-4o as default
        library = SmartLibrary("medical_library.json")
        llm_service = LLMService(provider="openai", model="gpt-4o")  # Use gpt-4o explicitly
        system_agent = SystemAgent(library, llm_service)
        workflow_processor = WorkflowProcessor(system_agent)
        workflow_generator = WorkflowGenerator(llm_service, library)
        
        # Seed the library with some initial records if empty
        if not library.records:
            # Add medical firmware
            await library.create_record(
                name="MedicalFirmware",
                record_type="FIRMWARE",
                domain="medical",
                description="Firmware for medical domain with HIPAA compliance",
                code_snippet=system_agent.firmware.get_firmware_prompt("medical")
            )
            
            # Add a basic symptom parser tool with hardcoded implementation
            await library.create_record(
                name="SymptomParser",
                record_type="TOOL",
                domain="medical",
                description="Parses patient symptoms into structured data",
                code_snippet=SYMPTOM_PARSER_CODE
            )
            
            logger.info("Initialized library with starter medical records")
        
        # Use a pre-defined workflow for testing
        print("Using pre-defined workflow for testing...")
        workflow_yaml = """
scenario_name: "Lupus Symptom Analysis"
domain: "medical"
description: "Analyze symptoms for potential lupus diagnosis"

additional_disclaimers:
  - "# MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice."
  - "Always consult with qualified healthcare providers."

steps:
  - type: "DEFINE"
    item_type: "TOOL"
    name: "LupusSymptomAnalyzer"
    from_existing_snippet: "SymptomParser"
    reuse_as_is: true
    description: "Analyzes symptoms to determine likelihood of lupus"

  - type: "CREATE"
    item_type: "TOOL"
    name: "LupusSymptomAnalyzer"

  - type: "EXECUTE"
    item_type: "TOOL"
    name: "LupusSymptomAnalyzer"
    user_input: "Patient has joint pain in hands, fatigue, and a butterfly-shaped rash on face."
"""
        
        print("\nWorkflow to execute:")
        print("=" * 80)
        print(workflow_yaml)
        print("=" * 80)
        
        # Process the workflow
        print("\nExecuting workflow...")
        results = await workflow_processor.process_workflow(workflow_yaml)
        
        # Print execution results
        print("\nExecution Results:")
        for i, step in enumerate(results.get("steps", [])):
            print(f"Step {i+1}: {step.get('message', 'No message')}")
            
            # Print result if available
            if "result" in step:
                print(f"  Result: {step['result']}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())