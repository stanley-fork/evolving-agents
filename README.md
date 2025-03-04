# Evolving Agents Framework

A production-grade framework for creating, managing, and evolving AI agents in a controlled environment. The framework allows you to generate complete agent workflows from natural language requirements and intelligently reuse or evolve existing components, with support for multiple agent frameworks.

## Features

- **Natural Language to Working Agents**: Convert requirements directly into executable workflows
- **Smart Library**: Central repository that learns from each execution
- **Firmware Injection**: Enforce domain-specific governance rules across all agents
- **Semantic Evolution**: Intelligently reuse or adapt existing agents and tools
- **YAML Workflows**: Generated workflows are human-readable and editable
- **Multi-Agent Orchestration**: Coordinate specialized agents for complex tasks
- **Multi-Framework Support**: Seamlessly integrate agents from different frameworks (BeeAI, OpenAI, etc.)
- **Provider Architecture**: Extensible design for adding support for new agent frameworks

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/evolving-agents-framework.git
cd evolving-agents-framework

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

This framework relies on:
- PyYAML
- OpenAI API (or another LLM provider of your choice)
- BeeAI Framework (integrated for agent capabilities)

## Quick Start

### Using a Custom YAML Workflow

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
    # Initialize the framework components
    library = SmartLibrary("library.json")
    llm = LLMService(provider="openai", model="gpt-4o")
    
    # Initialize provider registry with desired frameworks
    provider_registry = ProviderRegistry()
    provider_registry.register_provider(BeeAIProvider(llm))
    
    # Initialize system agent with provider registry
    system = SystemAgent(library, llm, provider_registry=provider_registry)
    processor = WorkflowProcessor(system)
    
    # Define a medical analysis workflow
    workflow_yaml = """
    scenario_name: "Lupus Symptom Analysis"
    domain: "medical"
    description: "Analyze symptoms for potential lupus diagnosis"
    
    additional_disclaimers:
      - "# MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice."
      - "Always consult with qualified healthcare providers."
    
    steps:
      - type: "DEFINE"
        item_type: "TOOL"
        name: "LupusSymptomAnalyzer"
        from_existing_snippet: "SymptomParser"
        reuse_as_is: true
        description: "Analyzes symptoms to determine likelihood of lupus"
    
      - type: "CREATE"
        item_type: "TOOL"
        name: "LupusSymptomAnalyzer"
    
      - type: "EXECUTE"
        item_type: "TOOL"
        name: "LupusSymptomAnalyzer"
        user_input: "Patient has joint pain in hands, fatigue, and a butterfly-shaped rash on face."
    """
    
    # Execute the workflow
    results = await processor.process_workflow(workflow_yaml)
    
    # View the results
    print("Execution results:")
    for step in results["steps"]:
        print(f"- {step.get('message', 'No message')}")
    
    # Print final result if available
    for step in reversed(results["steps"]):
        if "result" in step:
            print("\nAnalysis result:")
            print(step["result"])
            break

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: Smart Insulin Management System

Here's an example that creates an AI agent using the BeeAI framework to monitor glucose levels and adjust insulin dosage:

```python
# Define a healthcare workflow using BeeAI framework
workflow_yaml = """
scenario_name: "Smart Insulin Management System"
domain: "medical"
description: "Monitor glucose levels and automatically adjust insulin dosage"

additional_disclaimers:
  - "# MEDICAL_DISCLAIMER: This is not a substitute for professional medical advice."
  - "# Always consult with qualified healthcare providers for medical decisions."

steps:
  # Define tools
  - type: "DEFINE"
    item_type: "TOOL"
    name: "HealthDataMonitor"
    description: "Fetches health data from smartwatch and CGM"
    
  - type: "DEFINE"
    item_type: "TOOL"
    name: "InsulinDosageCalculator"
    description: "Calculates appropriate insulin dosage based on health metrics"
    
  - type: "DEFINE"
    item_type: "TOOL"
    name: "InsulinPumpController"
    description: "Controls the smart insulin pump to deliver insulin doses"
    
  # Define a BeeAI agent that uses these tools
  - type: "DEFINE"
    item_type: "AGENT"
    name: "InsulinManagementAgent"
    description: "Agent that monitors glucose and manages insulin dosing"
    framework: "beeai"  # Specify the framework to use
    required_tools:     # List tools the agent needs
      - "HealthDataMonitor"
      - "InsulinDosageCalculator"
      - "InsulinPumpController"
    
  # Create and execute the components
  - type: "CREATE"
    item_type: "TOOL"
    name: "HealthDataMonitor"
    
  - type: "CREATE"
    item_type: "TOOL"
    name: "InsulinDosageCalculator"
    
  - type: "CREATE"
    item_type: "TOOL"
    name: "InsulinPumpController"
    
  - type: "CREATE"
    item_type: "AGENT"
    name: "InsulinManagementAgent"
    config:
      memory_type: "token"  # Framework-specific configuration
    
  - type: "EXECUTE"
    item_type: "AGENT"
    name: "InsulinManagementAgent"
    user_input: "My glucose is 180 mg/dL, I've been sitting at my desk all morning"
    execution_config:       # Execution-specific configuration
      max_iterations: 15
      enable_observability: true
"""
```

## Core Components

### 1. Provider Architecture

The framework supports multiple agent frameworks through a provider system:

```python
# Register a new provider
provider_registry = ProviderRegistry()
provider_registry.register_provider(BeeAIProvider(llm_service))

# Create an agent with a specific framework
await agent_factory.create_agent(
    record=agent_record,
    tools=tools,
    firmware_content=firmware_content,
    config={"memory_type": "token"}  # Framework-specific config
)

# Get available frameworks
frameworks = agent_factory.get_available_frameworks()

# Get configuration schema for a framework
config_schema = agent_factory.get_agent_creation_schema("beeai")
```

### 2. Smart Library

The Smart Library serves as a central repository that stores and manages:

```python
# Store a tool in the library
await library.create_record(
    name="SymptomParser",
    record_type="TOOL",
    domain="medical",
    description="Parses patient symptoms into structured data",
    code_snippet=symptom_parser_code,
    metadata={"framework": "beeai"}  # Optional framework specification
)

# Find tools by domain
medical_tools = await library.find_records_by_domain("medical", "TOOL")

# Search for semantically similar tools
results = await library.semantic_search(
    query="Analyze lupus symptoms", 
    record_type="TOOL",
    domain="medical"
)
```

### 3. Firmware Injection

Firmware injects governance rules and constraints into all agents and tools:

```python
# Get firmware for a specific domain
firmware_content = system_agent.firmware.get_firmware_prompt("medical")

# Medical domain will include specific rules like:
"""
- Include medical disclaimers
- Ensure HIPAA compliance
- Protect patient confidentiality
- Require medical validation
"""
```

### 4. System Agent

The System Agent implements the decision logic for reuse, evolution, or creation:

```python
# Process a request using the decision logic
result = await system_agent.decide_and_act(
    request="I need a tool to identify lupus symptoms",
    domain="medical",
    record_type="TOOL"
)

# Execute the resulting item
if result["action"] in ["reuse", "evolve", "create"]:
    execution = await system_agent.execute_item(
        result["record"]["name"],
        "Patient has joint pain, fatigue, and butterfly rash"
    )
```

### 5. Workflow Processing

Process YAML workflows with simple step definitions:

```yaml
steps:
  - type: "DEFINE"    # Define a component, either new or evolved from existing
  - type: "CREATE"    # Instantiate the component in the environment  
  - type: "EXECUTE"   # Run the component with specific input
```

## Advanced Use Cases

### Creating a BeeAI Agent with Custom Tools

```python
# Define a BeeAI agent that uses custom tools
await library.create_record(
    name="DataAnalysisAgent",
    record_type="AGENT",
    domain="data_science",
    description="Agent that analyzes complex datasets",
    code_snippet=agent_code,
    metadata={
        "framework": "beeai",
        "required_tools": ["DataLoader", "StatisticalAnalyzer", "ChartGenerator"]
    }
)

# Create agent with specific framework configuration
await agent_factory.create_agent(
    record=agent_record,
    tools=tools,
    config={
        "memory_type": "token",
        "execution": {
            "max_iterations": 20,
            "max_retries_per_step": 3
        }
    }
)

# Execute with observability enabled
result = await agent_factory.execute_agent(
    "DataAnalysisAgent", 
    "Analyze this customer churn dataset and identify key factors",
    {
        "enable_observability": True,
        "max_iterations": 25
    }
)
```

### Defining Custom Domains and Firmware

You can extend the framework with custom domains:

```python
# Define finance domain firmware
finance_firmware = """
You are an AI agent operating under strict financial compliance rules:

1. REGULATORY COMPLIANCE:
- Never provide specific investment advice without disclaimers
- Adhere to SEC regulations
- Maintain transparency about hypothetical returns

2. DATA PRIVACY:
- Handle all financial information confidentially
- Do not retain PII
- Anonymize all examples

3. DISCLAIMERS:
- Always include disclaimers about financial risks
- State that past performance doesn't guarantee future results
"""

# Register in the library
await library.create_record(
    name="FinanceFirmware",
    record_type="FIRMWARE",
    domain="finance",
    description="Financial domain governance rules",
    code_snippet=finance_firmware
)
```

## Directory Structure

```
evolving-agents-framework/
├── evolving_agents/
│   ├── core/
│   │   ├── llm_service.py        # LLM interface with provider support
│   │   └── system_agent.py       # Core decision logic
│   ├── firmware/
│   │   └── firmware.py           # Governance rule injection
│   ├── providers/
│   │   ├── base.py               # Base provider interface
│   │   ├── registry.py           # Provider registry
│   │   └── beeai_provider.py     # BeeAI framework provider
│   ├── smart_library/
│   │   └── smart_library.py      # Repository for all components
│   ├── tools/
│   │   └── tool_factory.py       # Dynamic tool creation and execution
│   ├── agents/
│   │   └── agent_factory.py      # Agent creation and management
│   ├── workflow/
│   │   ├── workflow_generator.py # YAML workflow generation
│   │   └── workflow_processor.py # YAML workflow execution
│   └── utils/
│       └── embeddings.py         # Embedding utilities
├── examples/
│   ├── medical_diagnosis_example.py
│   └── smart_insulin_management_example.py
├── config/
│   └── firmware/                 # Domain-specific firmware
└── tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache v2.0

## Acknowledgements

- Matias Molinas and Ismael Faro for the original concept and architecture
- BeeAI framework for integrated agent capabilities