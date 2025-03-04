# examples/medical_diagnosis_example.py

import asyncio
import logging
import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Hardcoded tool implementation for reliability
SYMPTOM_PARSER_CODE = '''
# MEDICAL_DISCLAIMER: This tool is for informational purposes only and not a substitute for professional medical advice.

def parse_symptoms(input_text):
    """
    Parse symptoms from free text into structured data.
    
    Args:
        input_text: Patient description of symptoms
        
    Returns:
        Dictionary of structured symptom data
    """
    # Simple parsing logic
    text = input_text.lower()
    symptoms = []
    
    # Extract common symptoms
    if "joint pain" in text or "arthralgia" in text:
        symptoms.append({"name": "joint pain", "severity": "unknown"})
    if "fatigue" in text or "tired" in text:
        symptoms.append({"name": "fatigue", "severity": "unknown"})
    if "rash" in text:
        symptoms.append({"name": "skin rash", "severity": "unknown"})
    if "butterfly" in text or "malar" in text:
        symptoms.append({"name": "butterfly rash", "severity": "unknown", "location": "face"})
    if "fever" in text:
        symptoms.append({"name": "fever", "severity": "unknown"})
    if "photosensitivity" in text or "sensitive to light" in text or "sun sensitivity" in text:
        symptoms.append({"name": "photosensitivity", "severity": "unknown"})
    if "headache" in text:
        symptoms.append({"name": "headache", "severity": "unknown"})
    
    # Format the output
    output = {
        "symptoms": symptoms,
        "disclaimer": "This is an automated parsing of symptoms. Medical professionals should verify.",
        "source": "medical symptom parser"
    }
    
    # Determine if symptoms match lupus pattern
    lupus_indicators = ["joint pain", "fatigue", "butterfly rash", "photosensitivity", "fever"]
    matched_indicators = [s["name"] for s in symptoms if s["name"] in lupus_indicators or 
                         (s["name"] == "skin rash" and "butterfly" in text)]
    
    if len(matched_indicators) >= 2:
        output["possible_conditions"] = ["Lupus (SLE)"]
        output["recommendation"] = "Consult with a rheumatologist for proper evaluation."
    
    return output

# Call the function with the input and store the result
result = parse_symptoms(input)
'''

DISEASE_ANALYZER_CODE = '''
# MEDICAL_DISCLAIMER: This tool is for informational purposes only and not a substitute for professional medical advice.

def analyze_disease_likelihood(input_text):
    """
    Analyze symptoms to provide disease likelihood assessment.
    
    Args:
        input_text: Structured symptom data or patient description
        
    Returns:
        Disease likelihood assessment
    """
    # Parse the input
    try:
        # Check if input is already parsed symptoms in JSON format
        import json
        parsed_input = json.loads(input_text)
        symptoms = parsed_input.get("symptoms", [])
    except:
        # If not JSON, parse the raw text
        text = input_text.lower()
        symptoms = []
        
        if "joint pain" in text or "arthralgia" in text:
            symptoms.append({"name": "joint pain"})
        if "fatigue" in text or "tired" in text:
            symptoms.append({"name": "fatigue"})
        if "rash" in text:
            symptoms.append({"name": "skin rash"})
        if "butterfly" in text or "malar" in text:
            symptoms.append({"name": "butterfly rash", "location": "face"})
        if "fever" in text:
            symptoms.append({"name": "fever"})
        if "photosensitivity" in text or "sensitive to light" in text:
            symptoms.append({"name": "photosensitivity"})
    
    # Disease likelihood analysis
    diseases = []
    
    # Check for Lupus (SLE)
    lupus_indicators = ["joint pain", "fatigue", "butterfly rash", "photosensitivity", "fever"]
    lupus_matches = [s["name"] for s in symptoms if s["name"] in lupus_indicators]
    
    if len(lupus_matches) >= 3:
        lupus_likelihood = "high"
    elif len(lupus_matches) >= 2:
        lupus_likelihood = "moderate"
    elif len(lupus_matches) >= 1:
        lupus_likelihood = "low"
    else:
        lupus_likelihood = "very low"
    
    diseases.append({
        "name": "Systemic Lupus Erythematosus (SLE)",
        "likelihood": lupus_likelihood,
        "matched_symptoms": lupus_matches,
        "description": "An autoimmune disease that can affect multiple organ systems."
    })
    
    # Check for Rheumatoid Arthritis
    ra_indicators = ["joint pain", "fatigue", "fever"]
    ra_matches = [s["name"] for s in symptoms if s["name"] in ra_indicators]
    
    if "joint pain" in ra_matches and len(ra_matches) >= 2:
        ra_likelihood = "moderate"
    elif "joint pain" in ra_matches:
        ra_likelihood = "low"
    else:
        ra_likelihood = "very low"
    
    diseases.append({
        "name": "Rheumatoid Arthritis",
        "likelihood": ra_likelihood,
        "matched_symptoms": ra_matches,
        "description": "An autoimmune disease primarily affecting the joints."
    })
    
    # Format the results
    analysis = {
        "symptoms_analyzed": [s["name"] for s in symptoms],
        "diseases": diseases,
        "disclaimer": "This analysis is for informational purposes only and does not constitute medical advice or diagnosis.",
        "recommendation": "Please consult with a qualified healthcare provider for proper diagnosis and treatment."
    }
    
    return analysis

# Call the function with the input and store the result
result = analyze_disease_likelihood(input)
'''

MEDICAL_DIAGNOSIS_AGENT_CODE = '''
# MEDICAL_DISCLAIMER: This agent is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

def diagnose_medical_condition(input):
    """
    Agent that diagnoses potential medical conditions based on symptoms.
    
    This agent uses:
    1. SymptomParser tool to extract structured symptoms
    2. DiseaseAnalyzer tool to assess disease likelihood
    
    Args:
        input: Patient description or complaint
        
    Returns:
        Medical assessment and recommendations
    """
    # In a real BeeAI agent, this would use the actual tools
    # For this example, we'll simulate the agent's reasoning process
    
    assessment = {
        "input_summary": f"Patient reported: {input}",
        "symptom_analysis": "I'll analyze the symptoms described by extracting key medical indicators.",
        "diagnostic_process": "Based on the extracted symptoms, I'll evaluate potential conditions.",
        "recommendations": []
    }
    
    # This would represent the agent's internal thinking
    assessment["reasoning"] = (
        "1. First, I'll parse the symptoms described in the patient's input\n"
        "2. Then I'll analyze these symptoms to identify potential conditions\n"
        "3. I'll assess the likelihood of each condition based on symptom patterns\n"
        "4. Finally, I'll provide recommendations based on the analysis"
    )
    
    # Final recommendations would typically come from tools
    assessment["recommendations"] = [
        "Consult with a qualified healthcare provider for proper evaluation",
        "Maintain a symptom diary to track frequency and severity",
        "Avoid self-diagnosis as symptoms may have multiple causes"
    ]
    
    assessment["disclaimer"] = "MEDICAL DISCLAIMER: This assessment is for informational purposes only and does not constitute medical advice or diagnosis. Always consult with qualified healthcare professionals."
    
    return assessment

# Call the function with the input text
result = diagnose_medical_condition(input)
'''

async def main():
    try:
        # Initialize components with provider system
        library = SmartLibrary("medical_library.json")
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        # Initialize provider registry and register BeeAI provider
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        
        # Initialize system agent with provider registry
        system_agent = SystemAgent(
            smart_library=library, 
            llm_service=llm_service,
            provider_registry=provider_registry
        )
        
        # Initialize workflow components
        workflow_processor = WorkflowProcessor(system_agent)
        workflow_generator = WorkflowGenerator(llm_service, library)
        
        # Seed the library with some initial records if empty
        if not library.records:
            print("Initializing library with starter medical records...")
            
            # Add medical firmware
            await library.create_record(
                name="MedicalFirmware",
                record_type="FIRMWARE",
                domain="medical",
                description="Firmware for medical domain with HIPAA compliance",
                code_snippet=system_agent.firmware.get_firmware_prompt("medical")
            )
            
            # Add a basic symptom parser tool
            await library.create_record(
                name="SymptomParser",
                record_type="TOOL",
                domain="medical",
                description="Parses patient symptoms into structured data",
                code_snippet=SYMPTOM_PARSER_CODE
            )
            
            # Add disease analyzer tool
            await library.create_record(
                name="DiseaseAnalyzer",
                record_type="TOOL",
                domain="medical",
                description="Analyzes symptoms to determine disease likelihood",
                code_snippet=DISEASE_ANALYZER_CODE
            )
            
            # Add medical diagnosis agent using BeeAI framework
            await library.create_record(
                name="MedicalDiagnosisAgent",
                record_type="AGENT",
                domain="medical",
                description="Agent that diagnoses potential medical conditions",
                code_snippet=MEDICAL_DIAGNOSIS_AGENT_CODE,
                metadata={
                    "framework": "beeai",
                    "required_tools": ["SymptomParser", "DiseaseAnalyzer"]
                }
            )
            
            print("Library initialized with starter medical records!")
        
        # Define a lupus diagnosis workflow
        workflow_yaml = """
scenario_name: "Lupus Diagnosis Assistant"
domain: "medical"
description: "Analyze symptoms for potential lupus diagnosis and provide recommendations"

additional_disclaimers:
  - "# MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice."
  - "# Always consult with qualified healthcare providers for proper diagnosis and treatment."
  - "# In case of medical emergency, contact emergency services immediately."

steps:
  # Define a specialized lupus analyzer tool from existing symptom parser
  - type: "DEFINE"
    item_type: "TOOL"
    name: "LupusSymptomAnalyzer"
    from_existing_snippet: "SymptomParser"
    reuse_as_is: false
    evolve_changes:
      docstring_update: "Enhanced to specifically analyze lupus symptoms with higher sensitivity."
    description: "Analyzes symptoms to determine likelihood of lupus"

  # Create the lupus analyzer tool
  - type: "CREATE"
    item_type: "TOOL"
    name: "LupusSymptomAnalyzer"

  # Create the disease analyzer tool
  - type: "CREATE"
    item_type: "TOOL"
    name: "DiseaseAnalyzer"

  # Create the BeeAI medical diagnosis agent
  - type: "CREATE"
    item_type: "AGENT"
    name: "MedicalDiagnosisAgent"
    config:
      memory_type: "token"

  # Execute the lupus analyzer with a sample input
  - type: "EXECUTE"
    item_type: "TOOL"
    name: "LupusSymptomAnalyzer"
    user_input: "Patient has joint pain in hands, fatigue, and a butterfly-shaped rash on face. Also reports photosensitivity."

  # Execute the disease analyzer with the same symptoms
  - type: "EXECUTE"
    item_type: "TOOL"
    name: "DiseaseAnalyzer"
    user_input: "Patient has joint pain in hands, fatigue, and a butterfly-shaped rash on face. Also reports photosensitivity."

  # Execute the medical diagnosis agent
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "MedicalDiagnosisAgent"
    user_input: "I've been experiencing joint pain in my hands, constant fatigue, and noticed a rash across my cheeks and nose. I also seem to be more sensitive to sunlight lately."
    execution_config:
      max_iterations: 10
      enable_observability: true
"""
        
        print("\n" + "="*80)
        print("LUPUS DIAGNOSIS WORKFLOW:")
        print(workflow_yaml)
        print("="*80 + "\n")
        
        # Process the workflow
        print("Executing workflow...")
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
            
            # Print result if available
            if "result" in step:
                result = step["result"]
                if isinstance(result, str) and len(result) > 500:
                    # Truncate long string results
                    print(f"  Result: {result[:500]}... (truncated)")
                elif isinstance(result, dict):
                    # Pretty print dictionary results
                    print(f"  Result: {json.dumps(result, indent=2)}")
                else:
                    print(f"  Result: {result}")
            
            print("-" * 60)
        
        # Generate a new workflow from natural language
        print("\nGenerating a new workflow from natural language request...")
        
        nl_request = """
        Create a medical assistant that can analyze symptoms of rheumatoid arthritis
        and provide lifestyle recommendations.
        """
        
        generated_workflow = await workflow_generator.generate_workflow(
            requirements=nl_request,
            domain="medical"
        )
        
        print("\nGenerated Workflow:")
        print("="*80)
        print(generated_workflow)
        print("="*80)
        
        # Check available frameworks
        frameworks = system_agent.agent_factory.get_available_frameworks()
        print(f"\nAvailable agent frameworks: {frameworks}")
        
        # Test the BeeAI agent with additional scenarios directly
        print("\nTesting Medical Diagnosis Agent with Additional Scenarios:")
        print("="*80)
        
        test_scenarios = [
            "I've been having severe joint pain in my knees and hands, especially in the morning. The pain seems to be symmetrical, affecting both sides equally. I've also been feeling unusually tired.",
            "I've had a persistent headache for three days along with a stiff neck. I also have a slight fever and feel sensitive to bright lights."
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\nScenario {i+1}: {scenario}")
            
            # Use the agent_factory to execute the agent with the scenario
            result = await system_agent.agent_factory.execute_agent(
                "MedicalDiagnosisAgent", 
                scenario,
                {
                    "max_iterations": 5,
                    "enable_observability": True
                }
            )
            
            print(f"Agent Response:")
            if isinstance(result["result"], dict):
                print(json.dumps(result["result"], indent=2))
            else:
                print(result["result"])
            
            print("-" * 60)
        
        print("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())