# examples/simple_medical_diagnosis.py

import asyncio
import os
import sys
import logging
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.smart_library.record import LibraryRecord, RecordType, RecordStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize components
        library = SmartLibrary("library.json")
        llm = LLMService(provider="openai")
        system = SystemAgent(library, llm)
        
        # Load firmware
        await system.firmware_manager.load_firmware_from_yaml("config/firmware/base_firmware.yaml")
        
        # Manually create and add some basic tools to the library if they don't exist
        
        # Check if tools already exist
        general_symptom_parser = await library.find_record_by_name("General Symptom Parser")
        if not general_symptom_parser:
            symptom_parser_tool = LibraryRecord(
                name="General Symptom Parser",
                record_type=RecordType.TOOL,
                domain="medical",
                description="Tool to extract and parse symptoms from user input",
                code_snippet="""
                \"\"\"
                # MEDICAL_DISCLAIMER: This tool is for informational purposes only and not a substitute for professional medical advice.
                # Always consult with qualified healthcare providers.
                
                A general symptom parser that extracts and categorizes symptoms from user descriptions.
                
                This tool follows HIPAA compliance guidelines and medical best practices.
                \"\"\"
                
                def parse_symptoms(input):
                    \"\"\"
                    Parse symptoms from user input.
                    
                    Args:
                        input: Dictionary containing user's symptom description
                        
                    Returns:
                        Dictionary with structured symptom information
                    \"\"\"
                    text = input.get("text", "")
                    if not text:
                        return {"error": "No symptom description provided"}
                        
                    # In a real implementation, this would use NLP to extract symptoms
                    # For now, we'll return a simple structured response
                    return {
                        "parsed_symptoms": [
                            {"symptom": "joint pain", "severity": "moderate", "duration": "unknown"},
                            {"symptom": "fatigue", "severity": "moderate", "duration": "unknown"},
                            {"symptom": "skin rash", "severity": "mild", "duration": "unknown"}
                        ],
                        "disclaimer": "This is an automated parsing of symptoms. Medical professionals should verify."
                    }
                    
                # Entry point for the tool
                result = parse_symptoms(input)
                """,
                version="1.0.0",
                status=RecordStatus.ACTIVE,
                tags=["medical", "symptoms", "parser"]
            )
            await library.save_record(symptom_parser_tool)
            logger.info("Added General Symptom Parser to the library")
        
        decision_tree = await library.find_record_by_name("Decision Tree Template")
        if not decision_tree:
            decision_tree_tool = LibraryRecord(
                name="Decision Tree Template",
                record_type=RecordType.TOOL,
                domain="medical",
                description="A template for creating medical decision trees",
                code_snippet="""
                \"\"\"
                # MEDICAL_DISCLAIMER: This tool is for informational purposes only and not a substitute for professional medical advice.
                # Always consult with qualified healthcare providers.
                
                A decision tree template for medical assessments.
                
                This tool follows HIPAA compliance guidelines and medical best practices.
                \"\"\"
                
                def evaluate_decision_tree(input):
                    \"\"\"
                    Evaluate a set of inputs against a decision tree.
                    
                    Args:
                        input: Dictionary containing data to evaluate
                        
                    Returns:
                        Dictionary with assessment results
                    \"\"\"
                    symptoms = input.get("symptoms", [])
                    if not symptoms:
                        return {"error": "No symptoms provided for evaluation"}
                        
                    # In a real implementation, this would use a proper decision tree
                    # For now, we'll return a simple assessment
                    return {
                        "assessment": {
                            "condition": "Unspecified condition",
                            "likelihood": "Medium",
                            "confidence": 0.7,
                            "recommended_actions": ["Consult with a specialist", "Further testing"]
                        },
                        "disclaimer": "This is an automated assessment only. Medical professionals should make the final diagnosis."
                    }
                    
                # Entry point for the tool
                result = evaluate_decision_tree(input)
                """,
                version="1.0.0",
                status=RecordStatus.ACTIVE,
                tags=["medical", "decision tree", "template"]
            )
            await library.save_record(decision_tree_tool)
            logger.info("Added Decision Tree Template to the library")
        
        lupus_agent = await library.find_record_by_name("Lupus Assessment Agent")
        if not lupus_agent:
            lupus_agent = LibraryRecord(
                name="Lupus Assessment Agent",
                record_type=RecordType.AGENT,
                domain="medical",
                description="Agent that evaluates symptoms to determine the likelihood of lupus",
                code_snippet="""
                \"\"\"
                # MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice.
                # Always consult with qualified healthcare providers.
                # For informational purposes only.
                
                A lupus assessment agent that evaluates symptoms to determine the likelihood of lupus.
                
                This agent follows HIPAA compliance guidelines and medical best practices.
                \"\"\"
                
                from beeai_framework.agents.bee.agent import BeeAgent
                from beeai_framework.agents.types import AgentMeta
                
                class LupusAssessmentAgent(BeeAgent):
                    \"\"\"
                    A specialized agent for assessing the likelihood of lupus based on symptoms.
                    
                    This agent should always:
                    1. Include medical disclaimers
                    2. Avoid making definitive diagnoses
                    3. Suggest consulting healthcare professionals
                    4. Consider multiple possible explanations for symptoms
                    5. Recommend appropriate tests for confirmation
                    \"\"\"
                    
                    def __init__(self, llm, tools, memory):
                        \"\"\"Initialize the lupus assessment agent.\"\"\"
                        meta = AgentMeta(
                            name="Lupus Assessment Agent",
                            description="An agent that evaluates symptoms to determine the likelihood of lupus while maintaining HIPAA compliance.",
                            tools=tools
                        )
                        super().__init__(llm=llm, tools=tools, memory=memory, meta=meta)
                """,
                version="1.0.0",
                status=RecordStatus.ACTIVE,
                tags=["medical", "lupus", "assessment", "agent"]
            )
            await library.save_record(lupus_agent)
            logger.info("Added Lupus Assessment Agent to the library")
        
        # Let's create a custom workflow YAML instead of generating one
        custom_workflow = """
scenario_name: "Lupus Symptom Analysis Workflow"
domain: "medical"
description: "This workflow analyzes symptoms of joint pain, fatigue, and skin rash to assess the likelihood of lupus."

additional_disclaimers:
  - "This analysis is not a substitute for medical advice. Consult a healthcare professional for a proper diagnosis."
  - "Symptom analysis may not account for all unique patient conditions and variability."

steps:
  - type: "DEFINE"
    item_type: "TOOL"
    name: "Lupus Symptom Parser"
    from_existing_snippet: "General Symptom Parser"
    reuse_as_is: true
    description: "Parses and extracts lupus-related symptoms from user descriptions"

  - type: "CREATE"
    item_type: "TOOL"
    name: "Lupus Symptom Parser"

  - type: "EXECUTE"
    item_type: "TOOL"
    name: "Lupus Symptom Parser"
    user_input: "Patient has joint pain in hands and knees, chronic fatigue, and a butterfly-shaped rash across the face."

  - type: "DEFINE"
    item_type: "AGENT"
    name: "Lupus Assessment Agent"
    from_existing_snippet: "Lupus Assessment Agent"
    reuse_as_is: true
    description: "Evaluates symptoms to determine the likelihood of lupus"

  - type: "CREATE"
    item_type: "AGENT"
    name: "Lupus Assessment Agent"

  - type: "EXECUTE"
    item_type: "AGENT"
    name: "Lupus Assessment Agent"
    user_input: "Analyze these symptoms for lupus: joint pain in hands and knees, chronic fatigue, and a butterfly-shaped rash across the face."
"""
        
        # Save the custom workflow to a file
        workflow_path = "examples/lupus_workflow.yaml"
        with open(workflow_path, "w") as f:
            f.write(custom_workflow)
        
        logger.info(f"Created custom workflow at {workflow_path}")
        
        # Execute the custom workflow
        logger.info("Executing custom workflow...")
        result = await system.process_yaml_workflow(workflow_path)
        
        # Print the workflow
        print("\nCustom Workflow:")
        print("=" * 80)
        print(custom_workflow)
        print("=" * 80)
        
        # Print execution results
        print("\nExecution Results:")
        for i, step in enumerate(result["execution"]["steps"]):
            print(f"Step {i+1}: {step.get('message', 'No message')}")
            
        # Print final output if available
        for step in reversed(result["execution"]["steps"]):
            if "result" in step:
                print("\nFinal Result:")
                print("-" * 80)
                print(step["result"])
                print("-" * 80)
                break
        
        # Generate and execute a workflow from requirements
        logger.info("Processing natural language request...")
        nl_result = await system.process_request(
            request="Create an agent that can analyze symptoms of joint pain, fatigue, and skin rash to determine if they might indicate lupus.",
            domain="medical",
            output_yaml_path="examples/generated_workflow.yaml"
        )
        
        print("\nGenerated Workflow:")
        print("=" * 80)
        print(nl_result["workflow_yaml"])
        print("=" * 80)
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())