# Evolving Agents Framework

A production-grade framework for creating, managing, and evolving AI agents with agent-to-agent communication capabilities. The framework allows you to generate complete agent workflows from natural language requirements and intelligently reuse or evolve existing components, with support for multiple agent frameworks.

## Key Features

- **Agent-to-Agent Communication**: Enable specialized agents to collaborate on complex tasks through structured communication protocols
- **Semantic Evolution**: Intelligently reuse or adapt existing agents and tools based on semantic similarity
- **Smart Library with OpenAI Embeddings**: Powerful semantic search capabilities to find the most relevant components
- **Multi-Framework Support**: Seamlessly integrate agents from different frameworks (BeeAI, OpenAI, etc.)
- **Firmware Injection**: Enforce domain-specific governance rules across all agents
- **YAML Workflows**: Human-readable, version-controlled agent workflows

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evolving-agents-framework.git
cd evolving-agents-framework

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start: Using the System Agent's Decision Logic

This example demonstrates how the SystemAgent dynamically decides whether to reuse, evolve, or create components based on semantic similarity:

```python
import asyncio
import logging
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize components
    library = SmartLibrary("agent_library.json")
    llm = LLMService(provider="openai", model="gpt-4o")
    provider_registry = ProviderRegistry()
    provider_registry.register_provider(BeeAIProvider(llm))
    
    # Initialize the System Agent
    system = SystemAgent(library, llm, provider_registry=provider_registry)
    
    # Use the System Agent to process a request for an invoice processor
    print("Requesting an invoice processing agent...")
    result = await system.decide_and_act(
        request="I need an agent that can analyze invoices and extract total amounts",
        domain="document_processing",
        record_type="AGENT"
    )
    
    print(f"Decision: {result['action']}")
    print(f"Selected/Created: {result['record']['name']}")
    
    if 'similarity' in result:
        print(f"Similarity score: {result['similarity']:.2f}")
    
    # Execute the chosen agent with a sample invoice
    invoice_text = """
    INVOICE #12345
    Date: 2023-05-15
    Vendor: TechSupplies Inc.
    
    Items:
    1. Laptop Computer - $1,200.00 (2 units)
    2. Wireless Mouse - $25.00 (5 units)
    
    Subtotal: $1,680.00
    Tax (8.5%): $142.80
    Total Due: $1,822.80
    """
    
    # Execute the agent/tool that was selected or created
    execution_result = await system.execute_item(
        result['record']['name'], 
        invoice_text
    )
    
    print("\nExecution Result:")
    print(execution_result['result'])

if __name__ == "__main__":
    asyncio.run(main())
```

This example shows how the SystemAgent:
1. Decides whether to reuse an existing agent, evolve one, or create a new one
2. Makes this decision based on semantic similarity to existing components
3. Executes the chosen agent automatically

## Agent-to-Agent Communication Example

This example shows how agents can communicate with each other through a workflow:

```python
import asyncio
import logging
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.workflow.workflow_processor import WorkflowProcessor
from evolving_agents.providers.registry import ProviderRegistry
from evolving_agents.providers.beeai_provider import BeeAIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize components
    library = SmartLibrary("agent_library.json")
    llm = LLMService(provider="openai", model="gpt-4o")
    provider_registry = ProviderRegistry()
    provider_registry.register_provider(BeeAIProvider(llm))
    system = SystemAgent(library, llm, provider_registry=provider_registry)
    processor = WorkflowProcessor(system)
    
    # Define a document processing workflow
    workflow_yaml = """
    scenario_name: "Document Processing System"
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
    
      - type: "CREATE"
        item_type: "AGENT"
        name: "CoordinatorAgent"
        config:
          memory_type: "token"
    
      # Execute the workflow with an invoice document
      - type: "EXECUTE"
        item_type: "AGENT"
        name: "CoordinatorAgent"
        user_input: "Process this document: [Your document text here]"
        execution_config:
          max_iterations: 15
          enable_observability: true
    """
    
    # Execute the workflow
    results = await processor.process_workflow(workflow_yaml)
    
    # Print results
    for step in results["steps"]:
        print(f"- {step.get('message', 'No message')}")
        if "result" in step:
            print(f"\nResult:\n{step['result']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### System Agent Decision Logic

The System Agent implements the core decision logic for agent reuse or evolution:

```python
# If similarity >= 0.8: Reuse existing agent/tool
# If 0.4 <= similarity < 0.8: Evolve existing agent/tool
# If similarity < 0.4: Create new agent/tool

result = await system_agent.decide_and_act(
    request="I need a tool to extract invoice data",
    domain="document_processing",
    record_type="TOOL"
)

# Result will contain the action taken (reuse/evolve/create)
# and the resulting record
```

### Smart Library with Semantic Search

The Smart Library stores agents and tools, with powerful semantic search capabilities:

```python
# Search for semantically similar tools using OpenAI embeddings
results = await library.semantic_search(
    query="I need an agent that can understand and analyze documents",
    record_type="AGENT",
    threshold=0.3  # Only return results with similarity above 0.3
)

# Print search results
for record, score in results:
    print(f"Match: {record['name']} (Score: {score:.4f})")
    print(f"Description: {record['description']}")
```

### Agent Communication

Agents can communicate with each other through a communication tool:

```python
# Define a communication request
request = {
    "agent_name": "SpecialistAgent",
    "message": document_text,
    "data": {"document_type": "invoice"}
}

# Send the request using the AgentCommunicator tool
specialist_result = await agent_communicator.execute(json.dumps(request))
```

### Evolving Agents

Agents can evolve based on new requirements:

```python
# Define evolution workflow
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
      docstring_update: "Improved with enhanced invoice analysis capabilities"
    description: "Enhanced specialist that provides more detailed invoice analysis"

  # Create and execute the evolved agent
  - type: "CREATE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    config:
      memory_type: "token"
"""

# Process the evolution workflow
evolution_results = await processor.process_workflow(evolution_workflow)
```

## Use Cases

- **Document Processing Systems**: Coordinate multiple specialized agents to analyze, classify, and extract data from documents
- **Healthcare Workflows**: Medical agents communicating with pharmacy and insurance agents
- **Customer Service**: Routing agents communicating with specialized support agents
- **Financial Analysis**: Portfolio management agents communicating with market analysis agents

## Advanced Features

- **Firmware Injection**: Apply governance rules and constraints to all agents
- **Provider Architecture**: Easily extend to support new agent frameworks 
- **Workflow Generation**: Convert natural language requirements into executable workflows
- **Agent Evolution Tracking**: Track the lineage and evolution of agents over time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache v2.0

## Acknowledgements

- Matias Molinas and Ismael Faro for the original concept and architecture
- BeeAI framework for integrated agent capabilities