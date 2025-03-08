# examples/acp_integration_example.py

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
from evolving_agents.providers.acp_provider import ACPProvider
from evolving_agents.acp.client import ACPClient
from evolving_agents.acp.tools.document_analyzer import DocumentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample document for testing
SAMPLE_DOCUMENT = """
RESEARCH REPORT
Title: Advances in Artificial Intelligence
Date: 2023-05-20
Author: Dr. Jane Smith

Abstract:
This report examines recent developments in AI technology,
focusing on large language models and their applications in
business and science.

Key Findings:
1. Transformer-based models continue to dominate the field
2. Agent-based systems show particular promise for complex tasks
3. Ethical considerations remain a significant challenge
"""

async def main():
    try:
        # Initialize components
        library_path = "acp_example_library.json"
        
        # Create fresh library for this example
        if os.path.exists(library_path):
            os.remove(library_path)
            print(f"Created fresh library at {library_path}")
        
        # Initialize Smart Library and services
        library = SmartLibrary(library_path)
        llm_service = LLMService(provider="openai", model="gpt-4o")
        
        # Initialize ACP client
        acp_client = ACPClient(transport="memory")
        
        # Initialize provider registry with ACP provider
        provider_registry = ProviderRegistry()
        provider_registry.register_provider(BeeAIProvider(llm_service))
        provider_registry.register_provider(ACPProvider(llm_service, acp_client))
        
        # Initialize system agent
        system_agent = SystemAgent(
            smart_library=library,
            llm_service=llm_service,
            provider_registry=provider_registry
        )
        
        # Initialize workflow processor
        workflow_processor = WorkflowProcessor(system_agent)
        
        print("\n" + "="*80)
        print("DEMONSTRATING ACP INTEGRATION WITH EVOLVING AGENTS FRAMEWORK")
        print("="*80)
        
        # Step 1: Create a document analyzer tool
        print("\nStep 1: Creating document analyzer tool...")
        
        # Create a document analyzer tool record
        analyzer_tool = DocumentAnalyzer()
        tool_code = inspect.getsource(DocumentAnalyzer)
        
        analyzer_record = await library.create_record(
            name="DocumentAnalyzer",
            record_type="TOOL",
            domain="document_processing",
            description="Analyzes documents to identify type and extract key information",
            code_snippet=tool_code,
            tags=["analysis", "documents", "acp"],
            metadata={"framework": "acp"}
        )
        
        print(f"✓ Created DocumentAnalyzer tool (ID: {analyzer_record['id']})")
        
        # Step 2: Create ACP-enabled agents
        print("\nStep 2: Creating ACP-enabled agents...")
        
        # Create document analysis agent
        analysis_agent_result = await system_agent.decide_and_act(
            request="I need an agent that can analyze documents and extract key information",
            domain="document_processing",
            record_type="AGENT"
        )
        
        analysis_agent_name = analysis_agent_result['record']['name']
        print(f"✓ Created analysis agent: {analysis_agent_name}")
        
        # Create document summary agent
        summary_agent_result = await system_agent.decide_and_act(
            request="I need an agent that can summarize documents and highlight important points",
            domain="document_processing",
            record_type="AGENT"
        )
        
        summary_agent_name = summary_agent_result['record']['name']
        print(f"✓ Created summary agent: {summary_agent_name}")
        
        # Step 3: Register agents with ACP
        print("\nStep 3: Registering agents with ACP...")
        
        # Get agent instances
        analysis_agent = system_agent.active_items[analysis_agent_name]["instance"]
        summary_agent = system_agent.active_items[summary_agent_name]["instance"]
        
        # Register with ACP
        analysis_agent_id = await acp_client.register_agent(analysis_agent)
        summary_agent_id = await acp_client.register_agent(summary_agent)
        
        print(f"✓ Registered analysis agent with ACP (ID: {analysis_agent_id})")
        print(f"✓ Registered summary agent with ACP (ID: {summary_agent_id})")
        
        # Step 4: ACP Communication
        print("\nStep 4: Demonstrating ACP communication...")
        
        # Send a message from analysis agent to summary agent
        message = f"Please analyze and summarize this document: {SAMPLE_DOCUMENT}"
        
        response = await acp_client.send_message(
            sender_id=analysis_agent_id,
            recipient_id=summary_agent_id,
            message=message
        )
        
        print("\nACP Communication Result:")
        print("-" * 40)
        print(f"Sender: {analysis_agent_name} (ID: {analysis_agent_id})")
        print(f"Recipient: {summary_agent_name} (ID: {summary_agent_id})")
        print(f"Message: {message[:50]}...")
        print(f"Response: {response['content']}")
        
        # Step 5: View message history
        print("\nStep 5: Viewing ACP message history...")
        
        message_history = acp_client.get_message_history()
        
        print(f"\nMessage History ({len(message_history)} entries):")
        print("-" * 40)
        
        for i, entry in enumerate(message_history):
            print(f"Entry {i+1}:")
            print(f"  Timestamp: {entry['timestamp']}")
            print(f"  Direction: {entry['direction']}")
            
            if entry['direction'] == 'request':
                msg = entry['message']
                print(f"  Message Type: {msg.get('type', 'unknown')}")
                print(f"  Content: {str(msg.get('content', ''))[:50]}...")
            else:
                print(f"  Message: {str(entry['message'])[:50]}...")
            
            print()
        
        print("\nACP Integration Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())