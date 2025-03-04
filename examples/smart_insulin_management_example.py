# examples/smart_insulin_management_example.py

import asyncio
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider

# Import BeeAI framework components for reference
from beeai_framework.tools.tool import Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Mock microservice functions
def mock_smartwatch_data_service(params: str) -> Dict[str, Any]:
    """Mock service that provides smartwatch and CGM data."""
    # Parse parameters
    query_params = {}
    if params:
        query_params = {p.split('=')[0]: p.split('=')[1] for p in params.split('&')}
    
    # Generate mock data
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Base data
    data = {
        "timestamp": time_str,
        "device_id": "SW12345",
        "glucose_level": 110,  # mg/dL
        "heart_rate": 72,      # bpm
        "physical_activity": {
            "steps": 5200,
            "calories_burned": 320,
            "active_minutes": 45
        },
        "stress_level": 3,     # 1-10 scale
        "last_meal_time": (current_time.replace(hour=current_time.hour-3)).strftime("%Y-%m-%d %H:%M:%S"),
        "last_insulin_dose": {
            "time": (current_time.replace(hour=current_time.hour-3)).strftime("%Y-%m-%d %H:%M:%S"),
            "units": 4.5
        }
    }
    
    # Modify data based on time of day parameter if provided
    if "time_of_day" in query_params:
        tod = query_params["time_of_day"]
        if tod == "morning":
            data["glucose_level"] = 130  # Higher in morning
            data["heart_rate"] = 68
            data["physical_activity"]["steps"] = 800
        elif tod == "afternoon":
            data["glucose_level"] = 95   # Post-lunch dip
            data["heart_rate"] = 78
            data["physical_activity"]["steps"] = 3200
        elif tod == "evening":
            data["glucose_level"] = 145  # Evening rise
            data["heart_rate"] = 74
            data["physical_activity"]["steps"] = 6500
            data["stress_level"] = 5
    
    # Modify data based on activity level parameter if provided
    if "activity_level" in query_params:
        activity = query_params["activity_level"]
        if activity == "resting":
            data["heart_rate"] = 65
            data["physical_activity"]["steps"] = data["physical_activity"]["steps"] // 2
            data["physical_activity"]["active_minutes"] = 10
        elif activity == "active":
            data["heart_rate"] = 95
            data["physical_activity"]["steps"] = data["physical_activity"]["steps"] * 2
            data["physical_activity"]["active_minutes"] = 90
            data["glucose_level"] -= 15  # Activity lowers glucose
    
    return data

def mock_insulin_pump_service(dose_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock service that adjusts insulin dosage on the smart pump."""
    # Validate input
    required_fields = ["units", "delivery_speed", "auth_token"]
    for field in required_fields:
        if field not in dose_data:
            return {
                "status": "error",
                "message": f"Missing required field: {field}",
                "success": False
            }
    
    # Validate units
    units = float(dose_data["units"])
    if units < 0.5 or units > 20:
        return {
            "status": "error",
            "message": f"Invalid insulin units: {units}. Must be between 0.5 and 20.",
            "success": False
        }
    
    # Validate delivery speed
    speed = dose_data["delivery_speed"]
    if speed not in ["standard", "extended", "rapid"]:
        return {
            "status": "error",
            "message": f"Invalid delivery speed: {speed}. Must be standard, extended, or rapid.",
            "success": False
        }
    
    # Mock successful response
    return {
        "status": "success",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dose_id": "DOSE" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "units_delivered": units,
        "delivery_speed": speed,
        "message": f"Successfully delivered {units} units of insulin at {speed} speed.",
        "success": True
    }

# Hardcoded tool implementations for seeding the library
HEALTH_DATA_MONITOR_TOOL = '''
# Tool to fetch health data from smartwatch and CGM
import json

def fetch_health_data(input):
    """
    Fetches health data from smartwatch and continuous glucose monitor.
    
    Args:
        input: String containing parameters or user description
        
    Returns:
        Health data as structured JSON
    """
    try:
        # Parse input for parameters
        params = ""
        if "when" in input.lower():
            if "morning" in input.lower():
                params = "time_of_day=morning"
            elif "afternoon" in input.lower():
                params = "time_of_day=afternoon"
            elif "evening" in input.lower() or "night" in input.lower():
                params = "time_of_day=evening"
        
        if "resting" in input.lower() or "sitting" in input.lower() or "desk" in input.lower():
            params += "&activity_level=resting"
        elif "active" in input.lower() or "walking" in input.lower() or "running" in input.lower() or "jog" in input.lower():
            params += "&activity_level=active"
        
        # Call the mock service (in real implementation, this would be an API call)
        data = mock_smartwatch_data_service(params)
        
        # Add some context for better understanding
        if data["glucose_level"] > 180:
            data["glucose_status"] = "high"
        elif data["glucose_level"] < 70:
            data["glucose_status"] = "low"
        else:
            data["glucose_status"] = "normal"
            
        return data
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# Call the function with the input text
result = fetch_health_data(input)
'''

async def main():
    try:
        # Initialize components with proper provider system
        library = SmartLibrary("insulin_management_library.json")
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        # Initialize the provider registry and add the BeeAI provider
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        
        # Initialize system agent with the provider registry
        system_agent = SystemAgent(
            smart_library=library, 
            llm_service=llm_service,
            provider_registry=provider_registry
        )
        
        # Initialize workflow components
        workflow_processor = WorkflowProcessor(system_agent)
        workflow_generator = WorkflowGenerator(llm_service, library)
        
        # Seed the library with initial records if empty
        if not library.records:
            print("Initializing smart library with seed components...")
            
            # Add medical firmware
            await library.create_record(
                name="MedicalFirmware",
                record_type="FIRMWARE",
                domain="medical",
                description="Firmware for medical domain with HIPAA compliance",
                code_snippet=system_agent.firmware.get_firmware_prompt("medical")
            )
            
            # Add the existing health data monitor tool
            await library.create_record(
                name="HealthDataMonitor",
                record_type="TOOL",
                domain="medical",
                description="Tool to fetch health data from smartwatch and CGM",
                code_snippet=HEALTH_DATA_MONITOR_TOOL
            )
            
            print("Smart library initialized with seed components!")
        
        # Define our workflow with BeeAI framework specification
        workflow_yaml = """
scenario_name: "Smart Insulin Management System"
domain: "medical"
description: "Monitor glucose levels and automatically adjust insulin dosage"

additional_disclaimers:
  - "# MEDICAL_DISCLAIMER: This is not a substitute for professional medical advice."
  - "# Always consult with qualified healthcare providers for medical decisions."
  - "# In case of medical emergency, contact emergency services immediately."

steps:
  # Step 1: Define the insulin dosage calculator tool (created from scratch)
  - type: "DEFINE"
    item_type: "TOOL"
    name: "InsulinDosageCalculator"
    description: "Calculates appropriate insulin dosage based on health metrics"

  # Step 2: Define the insulin pump control tool (created from scratch)
  - type: "DEFINE"
    item_type: "TOOL"
    name: "InsulinPumpController"
    description: "Controls the smart insulin pump to deliver insulin doses"

  # Step 3: Define the insulin management agent with BeeAI framework
  - type: "DEFINE"
    item_type: "AGENT"
    name: "InsulinManagementAgent"
    description: "Agent that monitors glucose and manages insulin dosing using available tools"
    framework: "beeai"
    required_tools:
      - "HealthDataMonitor"
      - "InsulinDosageCalculator"
      - "InsulinPumpController"

  # Step 4: Create the insulin dosage calculator tool
  - type: "CREATE"
    item_type: "TOOL"
    name: "InsulinDosageCalculator"

  # Step 5: Create the insulin pump controller tool
  - type: "CREATE"
    item_type: "TOOL"
    name: "InsulinPumpController"

  # Step 6: Create the BeeAI insulin management agent
  - type: "CREATE"
    item_type: "AGENT"
    name: "InsulinManagementAgent"
    config:
      memory_type: "token"

  # Step 7: Test the insulin management agent
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "InsulinManagementAgent"
    user_input: "My glucose is 180 mg/dL, I've been sitting at my desk all morning"
    execution_config:
      max_iterations: 15
      enable_observability: true
"""
        
        print("\n" + "="*80)
        print("WORKFLOW TO EXECUTE:")
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
                if isinstance(step["result"], dict):
                    print(f"  Result: {json.dumps(step['result'], indent=2)}")
                else:
                    result_lines = step["result"].split("\n")
                    if len(result_lines) > 10:
                        # Truncate very long outputs
                        print(f"  Result: (showing first 10 lines)")
                        for line in result_lines[:10]:
                            print(f"    {line}")
                        print(f"    ... (truncated {len(result_lines) - 10} more lines)")
                    else:
                        print(f"  Result: {step['result']}")
            
            print("-" * 40)
        
        # Test the agent with additional scenarios
        print("\nTesting Insulin Management Agent with Additional Scenarios:")
        print("="*80)
        
        test_scenarios = [
            "My glucose is 95 mg/dL and dropping, and I'm about to start a 30-minute jog",
            "Glucose levels have been stable at 120 mg/dL, but I'm feeling stressed before a big meeting"
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"\nScenario {i+1}: {scenario}")
            
            # Use the agent_factory to execute the agent with the scenario
            result = await system_agent.agent_factory.execute_agent(
                "InsulinManagementAgent", 
                scenario,
                {
                    "max_iterations": 10,
                    "enable_observability": True
                }
            )
            
            if result["status"] == "success":
                print(f"Agent Response:\n{result['result']}")
            else:
                print(f"Error: {result['message']}")
            
            print("-" * 40)
        
        # Print available frameworks
        frameworks = system_agent.agent_factory.get_available_frameworks()
        print("\nAvailable frameworks:", frameworks)
        
        # Print BeeAI configuration schema
        beeai_schema = system_agent.agent_factory.get_agent_creation_schema("beeai")
        print("\nBeeAI configuration schema:")
        print(json.dumps(beeai_schema, indent=2))
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make mock services available to the tools
    globals()["mock_smartwatch_data_service"] = mock_smartwatch_data_service
    globals()["mock_insulin_pump_service"] = mock_insulin_pump_service
    
    asyncio.run(main())