# Evolving Agents Framework

A production-grade framework for creating, managing, and evolving AI agents with intelligent agent-to-agent communication. The framework enables you to build collaborative agent ecosystems that can semantically understand requirements, evolve based on past experiences, and communicate effectively to solve complex tasks.

![Evolving Agents](evolving-agents-logo.png)

## Key Features

- **Intelligent Agent Evolution**: Reuse, adapt, or create agents based on semantic similarity to existing components
- **Agent-to-Agent Communication**: Enable specialized agents to delegate tasks and collaborate on complex problems
- **Smart Library with Semantic Search**: Find the most relevant tools and agents using OpenAI embeddings
- **Self-improving System**: Agents get better over time through continuous evolution and learning
- **Human-readable YAML Workflows**: Define complex agent collaborations with simple, version-controlled YAML
- **Multi-Framework Support**: Seamlessly integrate agents from different frameworks (BeeAI, OpenAI, etc.)
- **Governance through Firmware**: Enforce domain-specific rules across all agents

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/evolving-agents-framework.git
cd evolving-agents-framework

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the comprehensive example
python examples/simplified_agent_communication.py
```

## Example: Agent Collaboration and Evolution

The framework lets you create agent ecosystems where specialized agents communicate and evolve:

```python
import asyncio
from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent

async def main():
    # Initialize the framework components
    library = SmartLibrary("agent_library.json")
    llm = LLMService(provider="openai", model="gpt-4o")
    system = SystemAgent(library, llm)
    
    # 1. Ask the System Agent for an invoice analysis agent
    # The System Agent will decide whether to reuse, evolve, or create one
    result = await system.decide_and_act(
        request="I need an agent that can analyze invoices and extract the total amount",
        domain="document_processing",
        record_type="AGENT"
    )
    
    print(f"Decision: {result['action']}")  # 'reuse', 'evolve', or 'create'
    print(f"Agent: {result['record']['name']}")
    
    # 2. Execute the agent with an invoice document
    invoice_text = """
    INVOICE #12345
    Date: 2023-05-15
    Vendor: TechSupplies Inc.
    Total Due: $1,822.80
    """
    
    execution = await system.execute_item(
        result['record']['name'], 
        invoice_text
    )
    
    print("\nInvoice Analysis Result:")
    print(execution["result"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Comprehensive Example Walkthrough

The framework includes a comprehensive example (`examples/simplified_agent_communication.py`) that demonstrates four key capabilities:

### 1. System Agent's Intelligent Decision Logic

The System Agent implements a sophisticated decision mechanism:
- If similarity ≥ 0.8: Reuse an existing agent/tool
- If 0.4 ≤ similarity < 0.8: Evolve an existing agent/tool 
- If similarity < 0.4: Create a new agent/tool

```python
# The System Agent dynamically decides what to do based on your request
invoice_agent_result = await system_agent.decide_and_act(
    request="I need an agent that can analyze invoices and extract the total amount",
    domain="document_processing",
    record_type="AGENT"
)

print(f"System Agent Decision: {invoice_agent_result['action']}")  # 'evolve'
print(f"Similarity Score: {invoice_agent_result['similarity']:.4f}")  # e.g., 0.4364
```

### 2. Agent-to-Agent Communication with Workflows

The framework enables specialized agents to communicate:

```python
# Define a workflow where agents communicate with each other
workflow_yaml = """
scenario_name: "Document Processing with Agent Communication"
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

  # Create the coordinator agent that will communicate with specialists
  - type: "CREATE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    config:
      memory_type: "token"

  # Execute the workflow with a document
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    user_input: "Process this document: {document_text}"
"""
```

In this workflow, the CoordinatorAgent uses the AgentCommunicator tool to delegate specialized tasks to the SpecialistAgent, demonstrating how agents can collaborate to solve complex problems.

### 3. Agent Evolution

Evolve existing agents to create enhanced versions:

```python
# Define an evolution workflow
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

  - type: "EXECUTE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    user_input: "{document_text}"
"""
```

This allows you to create specialized versions of agents that perform better on specific types of tasks.

### 4. Semantic Search with OpenAI Embeddings

Find semantically similar components in the library:

```python
# Search for agents that can process documents
search_results = await library.semantic_search(
    query="agent that can process and understand documents",
    record_type="AGENT",
    threshold=0.3  # Minimum similarity threshold
)

for record, score in search_results:
    print(f"Match: {record['name']} (Score: {score:.4f})")
    print(f"Description: {record['description']}")
```

## Core Components

### Smart Library

The central repository for agents, tools, and firmware:

```python
# Store a component in the library
await library.create_record(
    name="InvoiceAnalyzer",
    record_type="TOOL",
    domain="finance",
    description="Analyzes and extracts data from invoice documents",
    code_snippet=tool_code,
    tags=["invoice", "finance", "extraction"]
)

# Semantic search to find components
results = await library.semantic_search(
    query="tool that extracts data from invoices",
    record_type="TOOL",
    domain="finance"
)
```

### System Agent

The orchestrator that decides whether to reuse, evolve, or create components:

```python
# Process a request using the system agent
result = await system_agent.decide_and_act(
    request="I need a tool to analyze medical records",
    domain="healthcare",
    record_type="TOOL"
)

# Execute the resulting tool or agent
if result["action"] in ["reuse", "evolve", "create"]:
    execution = await system_agent.execute_item(
        result["record"]["name"],
        "Patient has high blood pressure and diabetes"
    )
```

### Workflow Processor

Process YAML-defined agent workflows:

```python
# Initialize workflow processor
processor = WorkflowProcessor(system_agent)

# Process a workflow 
results = await processor.process_workflow(workflow_yaml)
```

### Provider Architecture

Support for multiple agent frameworks:

```python
# Register providers
provider_registry = ProviderRegistry()
provider_registry.register_provider(BeeAIProvider(llm_service))
provider_registry.register_provider(OpenAIProvider(llm_service))

# Initialize system agent with providers
system = SystemAgent(library, llm, provider_registry=provider_registry)
```

## Use Cases

- **Document Processing**: Create specialized agents for different document types that collaborate to extract and analyze information
- **Healthcare**: Medical agents communicating with pharmacy and insurance agents to coordinate patient care
- **Financial Analysis**: Portfolio management agents collaborating with market analysis agents
- **Customer Service**: Routing agents delegating to specialized support agents
- **Multi-step Reasoning**: Break complex problems into components handled by specialized agents

## Advanced Features

- **Firmware Injection**: Enforce governance rules and constraints across all agents
- **Version Control**: Track the evolution of agents over time
- **Cross-domain Collaboration**: Enable agents from different domains to work together
- **Observability**: Monitor agent communications and decision processes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache v2.0

## Acknowledgements

- Matias Molinas and Ismael Faro for the original concept and architecture
- BeeAI framework for integrated agent capabilities