import asyncio
import os
import sys
import logging
import argparse

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.workflow.workflow_executor import WorkflowExecutor
from evolving_agents.agents.agent_factory import AgentFactory
from evolving_agents.tools.tool_factory import ToolFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    parser = argparse.ArgumentParser(description="Generate and execute a workflow from requirements")
    parser.add_argument("--requirements", type=str, help="Requirements file path")
    parser.add_argument("--requirements-text", type=str, help="Requirements text")
    parser.add_argument("--domain", type=str, default="general", help="Domain for the workflow")
    parser.add_argument("--library", type=str, default="smart_library.json", help="Smart Library path")
    parser.add_argument("--output", type=str, help="Output path for the generated workflow")
    parser.add_argument("--execute", action="store_true", help="Execute the generated workflow")
    args = parser.parse_args()
    
    # Load requirements
    if args.requirements:
        with open(args.requirements, 'r', encoding='utf-8') as f:
            requirements = f.read()
    elif args.requirements_text:
        requirements = args.requirements_text
    else:
        parser.error("Either --requirements or --requirements-text must be provided")
    
    # Initialize components
    smart_library = SmartLibrary(args.library)
    llm_service = LLMService(provider="openai")
    workflow_generator = WorkflowGenerator(llm_service, smart_library)
    
    # Generate workflow
    print(f"Generating workflow for domain: {args.domain}")
    workflow_yaml = await workflow_generator.generate_workflow(
        requirements=requirements,
        domain=args.domain,
        output_path=args.output
    )
    
    print("\nGenerated Workflow:")
    print("=" * 80)
    print(workflow_yaml)
    print("=" * 80)
    
    # Optionally execute the workflow
    if args.execute:
        print("\nExecuting workflow...")
        agent_factory = AgentFactory(smart_library, llm_service)
        tool_factory = ToolFactory(smart_library, llm_service)
        workflow_executor = WorkflowExecutor(smart_library, llm_service, agent_factory, tool_factory)
        
        results = await workflow_executor.execute_workflow(workflow_yaml)
        
        print("\nExecution Results:")
        print("=" * 80)
        for i, step_result in enumerate(results["steps"]):
            print(f"\nStep {i+1}:")
            print(f"Status: {step_result['status']}")
            print(f"Action: {step_result.get('action', 'N/A')}")
            print(f"Message: {step_result['message']}")
            
            if "result" in step_result:
                print("\nResult:")
                print("-" * 40)
                print(step_result["result"])
                print("-" * 40)
        
        print("\nWorkflow execution completed!")

if __name__ == "__main__":
    asyncio.run(main())