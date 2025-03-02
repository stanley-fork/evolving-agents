# Evolving Agents Framework

A production-grade framework for creating, managing, and evolving AI agents in a controlled environment. The framework allows you to generate complete agent workflows from natural language requirements and intelligently reuse or evolve existing components.

## Features

- **Natural Language to Working Agents**: Convert requirements directly into executable workflows
- **Smart Library**: Central repository that learns from each execution
- **Firmware Injection**: Enforce domain-specific governance rules across all agents
- **Semantic Evolution**: Intelligently reuse or adapt existing agents and tools
- **YAML Workflows**: Generated workflows are human-readable and editable
- **Multi-Agent Orchestration**: Coordinate specialized agents for complex tasks

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

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the framework components
    library = SmartLibrary("library.json")
    llm = LLMService(provider="openai", model="gpt-4o")
    system = SystemAgent(library, llm)
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

### Example Output

```
Execution results:
- Reused SymptomParser as LupusSymptomAnalyzer
- Created TOOL instance LupusSymptomAnalyzer
- Executed TOOL LupusSymptomAnalyzer

Analysis result:
{
  "symptoms": [
    {
      "name": "joint pain",
      "severity": "unknown"
    },
    {
      "name": "fatigue",
      "severity": "unknown"
    },
    {
      "name": "skin rash",
      "severity": "unknown"
    },
    {
      "name": "butterfly rash",
      "severity": "unknown",
      "location": "face"
    }
  ],
  "disclaimer": "This is an automated parsing of symptoms. Medical professionals should verify.",
  "possible_conditions": [
    "Lupus (SLE)"
  ],
  "recommendation": "Consult with a rheumatologist for proper evaluation."
}
```

## Core Components

### 1. Smart Library

The Smart Library serves as a central repository that stores and manages:

```python
# Store a tool in the library
await library.create_record(
    name="SymptomParser",
    record_type="TOOL",
    domain="medical",
    description="Parses patient symptoms into structured data",
    code_snippet=symptom_parser_code
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

### 2. Firmware Injection

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

### 3. System Agent

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

### 4. Workflow Processing

Process YAML workflows with simple step definitions:

```yaml
steps:
  - type: "DEFINE"    # Define a component, either new or evolved from existing
  - type: "CREATE"    # Instantiate the component in the environment  
  - type: "EXECUTE"   # Run the component with specific input
```

## Advanced Use Cases

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

### Tool Evolution Tracking

The framework automatically tracks usage metrics and evolution:

```python
# Each tool has:
# - usage_count
# - success_count
# - fail_count 
# - version
# - parent_id (if evolved from another tool)

# Example evolution
original_tool = await library.find_record_by_name("SymptomParser")
evolved_tool = await library.evolve_record(
    parent_id=original_tool["id"],
    new_code_snippet=updated_code,
    description="Enhanced symptom parser with better lupus detection"
)
```

## Directory Structure

```
evolving-agents-framework/
├── evolving_agents/
│   ├── core/
│   │   ├── llm_service.py        # LLM interface with BeeAI integration
│   │   └── system_agent.py       # Core decision logic
│   ├── firmware/
│   │   └── firmware.py           # Governance rule injection
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
│   └── medical_diagnosis_example.py
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