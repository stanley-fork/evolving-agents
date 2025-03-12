# Evolving Agents Framework

A toolkit for agent autonomy, evolution, and governance. Create agents that can understand requirements, evolve through experience, communicate effectively, and build new agents and tools - all while operating within governance guardrails.

![Evolving Agents](evolving-agents-logo.png)

## Why the World Needs This Toolkit
Current agent systems are designed primarily for humans to build and control AI agents. The Evolving Agents Ecosystem takes a fundamentally different approach: agents building agents.

Our toolkit provides:

- **Autonomous Evolution**: Agents learn from experience and improve themselves without human intervention
- **Agent Self-Discovery**: Agents discover and collaborate with other specialized agents to solve complex problems
- **Governance Firmware**: Enforceable guardrails that ensure agents evolve and operate within safe boundaries
- **Self-Building Systems**: The ability for agents to create new tools and agents when existing ones are insufficient
- **Agent-Centric Architecture**: Communication and capabilities built for agents themselves, not just their human creators

Instead of creating yet another agent framework, we build on existing frameworks like BeeAI to create a layer that enables agent autonomy, evolution, and self-governance - moving us closer to truly autonomous AI systems that improve themselves while staying within safe boundaries.

## Key Features

- **Intelligent Agent Evolution**: Reuse, adapt, or create agents based on semantic similarity to existing components
- **Agent-to-Agent Communication**: Enable specialized agents to delegate tasks and collaborate on complex problems
- **Smart Library with Semantic Search**: Find the most relevant tools and agents using OpenAI embeddings
- **Self-improving System**: Agents get better over time through continuous evolution and learning
- **Human-readable YAML Workflows**: Define complex agent collaborations with simple, version-controlled YAML
- **Multi-Framework Support**: Seamlessly integrate agents from different frameworks (BeeAI, OpenAI, etc.)
- **Governance through Firmware**: Enforce domain-specific rules across all agents
- **Service Bus Architecture**: Connect agents through a unified communication system with pluggable backends

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/evolving-agents-framework.git
cd evolving-agents-framework

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup the agent library
python examples/setup_simplified_agent_library.py

# Run the comprehensive example
python examples/simplified_agent_communication.py
```

## Agent-Centric Architecture

The Evolving Agents Framework is built on a true agent-centric architecture. The SystemAgent itself is implemented as a BeeAI ReActAgent that uses specialized tools to manage the agent ecosystem:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               SystemAgent                                    │
│                             (BeeAI ReActAgent)                               │
└────────────┬────────────────────────┬────────────────────────┬──────────────┘
             │                        │                        │
             ▼                        ▼                        ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│   SmartLibraryTools    │ │    ServiceBusTools     │ │   WorkflowTools        │
│ - SearchComponentTool  │ │ - RegisterProviderTool │ │ - ProcessWorkflowTool  │
│ - CreateComponentTool  │ │ - RequestServiceTool   │ │ - GenerateWorkflowTool │
│ - EvolveComponentTool  │ │ - DiscoverCapabilityTool│ │                       │
└────────────┬───────────┘ └────────────┬───────────┘ └────────────┬───────────┘
             │                          │                          │
             ▼                          ▼                          ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│     SmartLibrary       │ │       ServiceBus       │ │    WorkflowProcessor   │
│  (Storage Interface)   │ │  (Messaging Interface) │ │                        │
└────────────┬───────────┘ └────────────┬───────────┘ └────────────┬───────────┘
             │                          │                          │
             ▼                          ▼                          ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│   Storage Backends     │ │   Messaging Backends   │ │    YAML Definitions    │
│ - SimpleJSON           │ │ - SimpleJSON           │ │ - Agent Workflows      │
│ - VectorDB (future)    │ │ - Redis (future)       │ │ - Evolution Patterns   │
│ - Cloud (future)       │ │ - BeeAI ACP (future)   │ │ - Governance Rules     │
└────────────────────────┘ └────────────────────────┘ └────────────────────────┘
```

This architecture ensures that the system itself follows the "by agents, for agents" philosophy, with the SystemAgent making decisions and using specialized tools to interact with the underlying infrastructure.

### Smart Library Tools

The SmartLibrary provides three specialized tools for component management:

```
┌──────────────────────────────────────────────────────────┐
│                 Smart Library Tools                       │
└──────────┬─────────────────┬───────────────┬─────────────┘
           │                 │               │
           ▼                 ▼               ▼
┌────────────────┐ ┌────────────────┐ ┌───────────────────┐
│SearchComponent │ │CreateComponent │ │ EvolveComponent   │
│     Tool       │ │     Tool       │ │      Tool         │
│- search_by_    │ │- create_agent()│ │- evolve_agent()   │
│  query()       │ │- create_tool() │ │- evolve_tool()    │
│- search_by_    │ │- create_from   │ │- adapt_to_domain()│
│  similarity()  │ │  specification()│ │- improve_      │
│- find_by_      │ │                │ │  capability()    │
│  capability()  │ │                │ │                   │
└────────────────┘ └────────────────┘ └───────────────────┘
```

### Service Bus Tools

The Service Bus provides three specialized tools for inter-agent communication:

```
┌──────────────────────────────────────────────────────────┐
│                    Service Bus Tools                      │
└──────────┬─────────────────┬───────────────┬─────────────┘
           │                 │               │
           ▼                 ▼               ▼
┌────────────────┐ ┌────────────────┐ ┌───────────────────┐
│RegisterProvider│ │RequestService  │ │DiscoverCapability │
│     Tool       │ │     Tool       │ │      Tool         │
│- register()    │ │- request()     │ │- list_providers() │
│- update_       │ │- call_agent()  │ │- find_provider()  │
│  capabilities()│ │- send_message()│ │- get_capabilities()│
│- deregister()  │ │                │ │- search_           │
│                │ │                │ │  capabilities()    │
└────────────────┘ └────────────────┘ └───────────────────┘
```

### Service Bus: Communication Layer

The Service Bus provides a unified communication layer with pluggable backends:

```
┌──────────────────────────────────────┐
│          ServiceBusInterface         │
│ - register_provider()                │
│ - request_service()                  │
│ - send_message()                     │
│ - find_provider_for_capability()     │
└──────────────────┬───────────────────┘
                   │
        ┌──────────┴────────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌──────────────────┐    ┌───────────────┐
│ SimpleBackend │    │  RedisBackend    │    │ BeeAIBackend  │
│(JSON storage) │    │(High performance)│    │   (Future)    │
└───────────────┘    └──────────────────┘    └───────────────┘
```

### Smart Library: Component Storage

The Smart Library stores agent and tool definitions with pluggable storage backends:

```
┌───────────────────────────────────────┐
│         SmartLibraryInterface         │
│ - create_record()                     │
│ - semantic_search()                   │
│ - evolve_record()                     │
└───────────────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌──────────────────┐    ┌───────────────┐
│SimpleStorage  │    │ VectorDBStorage  │    │  CloudStorage │
│(JSON-based)   │    │(Embedding search)│    │  (Scalable)   │
└───────────────┘    └──────────────────┘    └───────────────┘
```

### Agent-Centric Communication Flow

The Service Bus enables powerful capability-based communication between agents:

```
┌──────────────┐                      ┌───────────────┐
│   Agent A    │                      │  Service Bus  │
│ (Requester)  │                      │               │
└──────┬───────┘                      └───────┬───────┘
       │                                      │
       │ 1. Request service by capability     │
       │ "I need document analysis"           │
       │─────────────────────────────────────▶│
       │                                      │
       │                                      │ 2. Find provider
       │                                      │    with capability
       │                                      │◀─────────────────▶┐
       │                                      │                   │
       │                                      │                  ┌┴───────────────┐
       │                                      │                  │  Capability    │
       │                                      │                  │   Registry     │
       │                                      │                  └────────────────┘
       │                                      │
       │                  ┌───────────────────┘
       │                  │
       │                  │ 3. Route request to best provider
       │                  │    for the requested capability
       │                  ▼
┌──────┴───────┐   ┌──────────────┐
│   Agent B    │◀──│   Agent C    │
│(Not selected)│   │  (Selected)  │
└──────────────┘   └──────┬───────┘
                          │
                          │ 4. Process request and return result
                          │
                          ▼
┌──────────────┐   ┌──────────────┐
│   Agent A    │◀──│  Service Bus │
│ (Requester)  │   │              │
└──────────────┘   └──────────────┘
```

## Example: Agent Collaboration via Service Bus

```python
# Initialize the SystemAgent with LLM
llm_service = LLMService(provider="openai", model="gpt-4o")
system_agent = SystemAgent(llm_service)

# The SystemAgent is a BeeAI ReActAgent that can:
# 1. Interact with the SmartLibrary through specialized library tools
# 2. Manage the Service Bus through specialized service bus tools
# 3. Create and evolve agents as needed

# Initialize tools for the SystemAgent
await system_agent.initialize_tools()

# Ask the SystemAgent to set up a document processing workflow
response = await system_agent.run("""
Please set up a document processing workflow with these components:
1. A document analyzer tool that can identify document types
2. A data extraction agent that can extract structured data from documents
3. A summarization agent that can create concise summaries
Then process this invoice: 
INVOICE #12345
Date: 2023-05-15
Vendor: TechSupplies Inc.
Total Due: $1,822.80
""")

# Behind the scenes, the SystemAgent:
# 1. Uses SearchComponentTool to find relevant components
# 2. Uses CreateComponentTool or EvolveComponentTool as needed
# 3. Uses RegisterProviderTool to register components with the Service Bus
# 4. Uses RequestServiceTool to orchestrate the document processing workflow
# 5. Returns the final results

print(response.result.text)
```

## Understanding the Comprehensive Example

The framework includes a comprehensive example (`examples/simplified_agent_communication.py`) that demonstrates four key capabilities. This example shows in detail how the Evolving Agents Framework creates, manages, and evolves AI agents to handle real-world tasks.

### Step 1: Setting Up the Agent Library

Before running the main example, we need to set up an initial library of agents and tools:

```bash
python examples/setup_simplified_agent_library.py
```

This script creates a foundation of BeeAI-compatible agents and tools:

- **DocumentAnalyzer Tool**: A real BeeAI tool that uses an LLM to analyze documents and identify their type
- **AgentCommunicator Tool**: A real BeeAI tool that facilitates communication between agents
- **SpecialistAgent**: A BeeAI agent specialized in detailed document analysis
- **CoordinatorAgent**: A BeeAI agent that orchestrates the document processing workflow

### Step 2: Demonstrating the System Agent's Decision Logic

The first part of the example demonstrates how the System Agent intelligently decides whether to reuse, evolve, or create a new agent based on semantic similarity:

```python
# The System Agent dynamically decides what to do based on your request
invoice_agent_result = await system_agent.decide_and_act(
    request="I need an agent that can analyze invoices and extract the total amount",
    domain="document_processing",
    record_type="AGENT"
)

print(f"System Agent Decision: {invoice_agent_result['action']}")  # 'create', 'evolve', or 'reuse'
```

The System Agent implements a sophisticated decision mechanism:
- If similarity ≥ 0.8: Reuse an existing agent/tool
- If 0.4 ≤ similarity < 0.8: Evolve an existing agent/tool 
- If similarity < 0.4: Create a new agent/tool

In the example, when we ask for an invoice analysis agent, it creates a new one. When we ask for a medical record analyzer, it evolves the existing SpecialistAgent (since it has a similarity score of around 0.46).

### Step 3: Agent-to-Agent Communication with Workflows

The second part demonstrates how agents communicate with each other through workflows defined in YAML:

```yaml
# A workflow where agents communicate with each other
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

  # Create the coordinator agent with the tools
  - type: "CREATE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    config:
      memory_type: "token"

  # Execute with an invoice document
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "CoordinatorAgent"
    user_input: "Process this document: {invoice}"
"""
```

In this workflow, the CoordinatorAgent delegates specialized tasks to the SpecialistAgent through the AgentCommunicator tool. The example shows a complete workflow execution for both an invoice and a medical record.

### Step 4: Agent Evolution

The third part demonstrates how agents can be evolved to create enhanced versions:

```yaml
# Evolution workflow
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
      docstring_update: "Improved with enhanced invoice analysis capabilities including line item detection"
    description: "Enhanced specialist that provides more detailed invoice analysis with line item extraction"

  # Create and execute the evolved agent
  - type: "CREATE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    config:
      memory_type: "token"

  # Test the evolved agent with an invoice
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "EnhancedInvoiceSpecialist"
    user_input: "{invoice}"
"""
```

This evolution process takes an existing agent (SpecialistAgent) and creates an enhanced version (EnhancedInvoiceSpecialist) with improved capabilities specific to invoice analysis.

### Step 5: Semantic Search with OpenAI Embeddings

The final part demonstrates how to find semantically similar components in the library:

```python
# Search for document processing agents
search_results = await library.semantic_search(
    query="agent that can process and understand documents",
    record_type="AGENT",
    threshold=0.3
)
```

This shows how the framework uses OpenAI embeddings to find the most relevant agents for a given task, allowing you to discover and reuse existing components based on their semantic meaning rather than just exact keyword matches.

## Service Bus and Capability-Based Communication

The Service Bus enables agents to communicate based on capabilities rather than explicit agent references:

```python
# Initialize the Service Bus with simple JSON backend
await system_agent.initialize_service_bus(backend_type="simple")

# Register components with capabilities using RegisterProviderTool
document_analyzer_id = await system_agent.tools["register_provider"].run(
    name="DocumentAnalyzer",
    capabilities=[{
        "id": "analyze_document",
        "name": "Document Analysis",
        "description": "Analyzes documents to determine their type",
        "confidence": 0.9
    }]
)

# Process a document using capability-based routing with RequestServiceTool
document_text = "INVOICE #12345\nDate: 2023-05-15\nTotal: $1,822.80"
analysis_response = await system_agent.tools["request_service"].run(
    capability="analyze_document",
    content=document_text
)

# Discover available capabilities with DiscoverCapabilityTool
available_analysis_tools = await system_agent.tools["discover_capability"].run(
    capability_type="document_analysis"
)

# The Service Bus automatically finds the most appropriate provider
print(f"Document type: {analysis_response['content']['document_type']}")
```

This capability-based communication enables:
1. **Dynamic Discovery**: Agents can find services without knowing provider details
2. **Loose Coupling**: Agents depend on capabilities, not specific implementations
3. **Graceful Evolution**: Providers can be upgraded without disrupting consumers
4. **Automatic Routing**: Requests are sent to the most appropriate provider

## Core Components

### SystemAgent as a BeeAI ReActAgent

The SystemAgent is implemented as a BeeAI ReActAgent with specialized tools:

### Smart Library Tools

The Smart Library tools allow the SystemAgent to manage components in the library:

```python
class SearchComponentTool(Tool):
    """Tool for searching components in the Smart Library."""
    
    name = "SearchComponentTool"
    description = "Search for agents and tools by query, similarity, or capability"
    
    # Tool implementation...
    async def search_by_query(self, query, record_type=None, domain=None, limit=5):
        """Search by natural language query."""
        # Implementation...

class CreateComponentTool(Tool):
    """Tool for creating components in the Smart Library."""
    
    name = "CreateComponentTool"
    description = "Create new agents and tools from specifications"
    
    # Tool implementation...
    async def create_agent(self, name, description, domain, code=None):
        """Create a new agent from specification."""
        # Implementation...

class EvolveComponentTool(Tool):
    """Tool for evolving components in the Smart Library."""
    
    name = "EvolveComponentTool"
    description = "Evolve existing agents and tools for new requirements"
    
    # Tool implementation...
    async def evolve_agent(self, agent_id, changes, new_requirements):
        """Evolve an existing agent with new capabilities."""
        # Implementation...
```

### Service Bus Tools

The Service Bus tools enable the SystemAgent to manage inter-agent communication:

```python
class RegisterProviderTool(Tool):
    """Tool for registering providers with the Service Bus."""
    
    name = "RegisterProviderTool"
    description = "Register agents and tools as service providers"
    
    # Tool implementation...
    async def register(self, name, capabilities):
        """Register a provider with its capabilities."""
        # Implementation...

class RequestServiceTool(Tool):
    """Tool for requesting services through the Service Bus."""
    
    name = "RequestServiceTool"
    description = "Request services by capability"
    
    # Tool implementation...
    async def request(self, capability, content):
        """Request a service by capability."""
        # Implementation...

class DiscoverCapabilityTool(Tool):
    """Tool for discovering capabilities in the Service Bus."""
    
    name = "DiscoverCapabilityTool"
    description = "Discover available capabilities and providers"
    
    # Tool implementation...
    async def find_provider(self, capability):
        """Find a provider for a specific capability."""
        # Implementation...
```

### Workflow Tools

Tools for workflow management:

```python
class ProcessWorkflowTool(Tool):
    """Tool for processing workflows."""
    
    name = "ProcessWorkflowTool"
    description = "Define and execute workflows in YAML format"
    
    # Tool implementation...
    
class GenerateWorkflowTool(Tool):
    """Tool for generating workflows from natural language."""
    
    name = "GenerateWorkflowTool"
    description = "Generate workflow YAML from requirements"
    
    # Tool implementation...
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

[Apache v2.0](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=matiasmolinas/evolving-agents&type=Timeline)](https://star-history.com/#matiasmolinas/evolving-agents&Timeline)

## Acknowledgements

- [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro) for the original concept and architecture
- BeeAI framework for integrated agent capabilities

## Final Note:
The code is currently being actively refactored to align with the README.md Some features described here may not yet be implemented or fully functional. The updated version reflecting this documentation will be available in the next few days. Stay tuned! 🚀