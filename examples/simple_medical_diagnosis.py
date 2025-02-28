# examples/simple_medical_diagnosis.py

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

logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize components
        library = SmartLibrary("library.json")
        llm = LLMService(provider="openai")
        system = SystemAgent(library, llm)
        
        # Load firmware
        await system.firmware_manager.load_firmware_from_yaml("config/firmware/base_firmware.yaml")
        
        # Load initial tools and agents
        with open("config/initial_records/base_tools.yaml", 'r') as f:
            yaml_content = f.read()
            
        # Process natural language request
        logger.info("Processing natural language request...")
        
        result = await system.process_request(
            request="I need an agent that can analyze symptoms of joint pain, fatigue, and skin rash to determine if they might indicate lupus.",
            domain="medical"
        )
        
        # Print the generated workflow
        print("\nGenerated Workflow:")
        print("=" * 80)
        print(result["workflow_yaml"])
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
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())