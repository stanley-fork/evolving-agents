# examples/openmeteo_system_agent_example.py

import asyncio
import logging
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BeeAI Framework imports
from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig, AgentMeta
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Evolving Agents Framework imports
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_library():
    """Set up a library with a reference to OpenMeteoTool."""
    library_path = "openmeteo_library.json"
    
    # Delete existing library if it exists
    if os.path.exists(library_path):
        os.remove(library_path)
        print(f"Removed existing library at {library_path}")
    
    # Create new library
    library = SmartLibrary(library_path)
    
    # Create a reference to the OpenMeteoTool
    openmeteo_code = """
    # BeeAI OpenMeteoTool Reference
    # This is a reference to the OpenMeteoTool from the BeeAI framework
    
    from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
    
    def get_weather(input):
        # This function wraps the OpenMeteoTool functionality
        # Parse the input as JSON with location details
        try:
            # Try to parse JSON input
            import json
            data = json.loads(input)
            location = data.get("location", "")
            country = data.get("country", None)
        except:
            # If not JSON, assume it's just a location name
            location = input
            country = None
        
        # In the actual implementation, we create and use the OpenMeteoTool
        # Here we just return a placeholder
        result = f"Weather data for {location}"
        if country:
            result += f", {country}"
        
        return result
    
    # Process input and return
    result = get_weather(input)
    """
    
    await library.create_record(
        name="OpenMeteoWeatherTool",
        record_type="TOOL",
        domain="weather",
        description="Gets current weather information using OpenMeteo API",
        code_snippet=openmeteo_code,
        metadata={
            "source": "beeai_framework.tools.weather.openmeteo",
            "class": "OpenMeteoTool"
        },
        tags=["weather", "tool", "openmeteo", "api"]
    )
    print("✓ Created OpenMeteoWeatherTool reference in library")
    
    return library

async def main():
    try:
        # 1. Set up the library with our tool reference
        print("\n=== SETTING UP LIBRARY ===")
        library = await setup_library()
        
        # 2. Initialize the LLM service and system agent
        print("\n=== INITIALIZING SYSTEM COMPONENTS ===")
        llm_service = LLMService(provider="openai", model="gpt-4o-mini")
        system_agent = SystemAgent(library, llm_service)
        print("✓ System Agent initialized")
        
        # 3. Use the system agent to find the weather tool
        print("\n=== FINDING OPENMETEO TOOL WITH SYSTEM AGENT ===")
        tool_result = await system_agent.decide_and_act(
            request="I need a tool that can get weather information from APIs",
            domain="weather",
            record_type="TOOL"
        )
        
        print(f"System Agent Decision: {tool_result['action']}")
        print(f"Tool: {tool_result['record']['name']}")
        
        # 4. Extract the class and source information
        found_tool_record = tool_result['record']
        
        tool_metadata = found_tool_record.get('metadata', {})
        tool_source = tool_metadata.get('source')
        tool_class = tool_metadata.get('class')
        
        print(f"\nExtracted Tool Information:")
        print(f"- Source: {tool_source}")
        print(f"- Class: {tool_class}")
        
        # 5. Create a real OpenMeteoTool instance based on the reference
        print("\n=== CREATING REAL OPENMETEO TOOL ===")
        openmeteo_tool = OpenMeteoTool()
        print("✓ Created real OpenMeteoTool instance")
        
        # 6. Create a BeeAgent with the tool
        print("\n=== CREATING BEEAGENT WITH OPENMETEO TOOL ===")
        
        # Create an OpenAI Chat Model
        chat_model = OpenAIChatModel(
            model_id="gpt-4o-mini",
            settings={}
        )
        
        # Create agent meta information
        agent_meta = AgentMeta(
            name="WeatherAgent",
            description="A weather agent that provides current weather conditions for any location",
            tools=[openmeteo_tool]
        )
        
        # Create the BeeAgent instance
        bee_agent = BeeAgent(
            llm=chat_model,
            tools=[openmeteo_tool],
            memory=TokenMemory(chat_model),
            meta=agent_meta
        )
        
        tools_count = len(bee_agent.input.tools)
        print(f"✓ BeeAgent created with {tools_count} tools")
        
        # 7. Interactive testing with the BeeAgent
        print("\n=== INTERACTIVE BEEAGENT TESTING ===")
        print("Ask about the weather in any location, or type 'exit' to quit")
        
        while True:
            query = input("\nYou: ")
            
            if query.lower() in ["exit", "quit"]:
                break
            
            try:
                # Create execution config
                exec_config = BeeAgentExecutionConfig(
                    max_retries_per_step=2,
                    total_max_retries=5,
                    max_iterations=10
                )
                
                # Execute the BeeAgent
                print("\nProcessing...")
                response = await bee_agent.run(
                    prompt=query,
                    execution=exec_config
                )
                
                print(f"\nAgent: {response.result.text}")
            except Exception as e:
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())