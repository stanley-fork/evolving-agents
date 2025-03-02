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
- OpenAI (or another LLM provider of your choice)

## Quick Start

### From Natural Language to Working Agent

```python
import asyncio
import logging

from evolving_agents.smart_library.smart_library import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent
from evolving_agents.workflow.workflow_generator import WorkflowGenerator
from evolving_agents.workflow.workflow_processor import WorkflowProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the framework components
    library = SmartLibrary("library.json")
    llm = LLMService()
    system = SystemAgent(library, llm)
    generator = WorkflowGenerator(llm, library)
    processor = WorkflowProcessor(system)
    
    # Generate a workflow from natural language
    workflow_yaml = await generator.generate_workflow(
        requirements="Create an agent that can analyze a patient's symptoms, provide a preliminary diagnosis for lupus, and research the latest immunosuppressant treatments. Include medical disclaimers.",
        domain="medical"
    )
    
    # Execute the workflow
    results = await processor.process_workflow(workflow_yaml)
    
    # View the results
    print(f"Generated workflow:\n{workflow_yaml}\n")
    print("Execution results:")
    for step in results["steps"]:
        print(f"- {step.get('message', 'No message')}")
    
    # Print final result if available
    for step in reversed(results["steps"]):
        if "result" in step:
            print("\nAgent response:")
            print(step["result"])
            break

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Output

```
Generated workflow:
scenario_name: "Lupus Diagnosis and Treatment Research"
domain: "medical"
description: "Analyzes symptoms, provides preliminary lupus diagnosis, and researches immunosuppressant treatments"

additional_disclaimers:
  - "# MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice."
  - "Always consult with qualified healthcare providers."
  - "For informational purposes only."

steps:
  - type: "DEFINE"
    item_type: "TOOL"
    name: "LupusDiagnosisTool"
    from_existing_snippet: "SymptomParser"
    evolve_changes:
      docstring_update: "Specialized for lupus diagnosis based on symptoms"
    description: "Analyzes patient symptoms to determine likelihood of lupus"

  - type: "CREATE"
    item_type: "TOOL"
    name: "LupusDiagnosisTool"

  - type: "EXECUTE"
    item_type: "TOOL"
    name: "LupusDiagnosisTool"
    user_input: "My patient has fatigue, joint pain, skin rash, and fever. What's the likelihood of lupus and what are the latest treatment options?"

Execution results:
- Evolved SymptomParser to LupusDiagnosisTool, injecting firmware for domain 'medical'.
- Created TOOL instance LupusDiagnosisTool
- Executed TOOL LupusDiagnosisTool

Agent response:
# MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice.
# Always consult with qualified healthcare providers.
# For informational purposes only.

## Preliminary Diagnosis

Based on the symptoms described (fatigue, joint pain, skin rash, and fever), there is a **moderate to high likelihood** of systemic lupus erythematosus (SLE). These symptoms represent classic signs of lupus:

- Fatigue: Present in over 90% of lupus patients
- Joint pain (arthralgia): Affects approximately 95% of patients
- Skin rash: The butterfly (malar) rash is characteristic of lupus
- Fever: Common during disease flares

**Additional testing recommended:**
- Antinuclear antibody (ANA) test
- Anti-double-stranded DNA (anti-dsDNA) antibody test
- Complete blood count (CBC)
- Erythrocyte sedimentation rate (ESR)
- C-reactive protein (CRP)
- Complement levels (C3 and C4)

## Latest Immunosuppressant Treatments

Recent advances in lupus treatment include:

1. **Belimumab (Benlysta)** - FDA-approved monoclonal antibody
2. **Anifrolumab (Saphnelo)** - FDA-approved in 2021
3. **Voclosporin (Lupkynis)** - Approved January 2021
4. **JAK inhibitors** (Baricitinib, Tofacitinib) - Under investigation

Please consult with a rheumatologist for appropriate treatment options.
```

## More Advanced Use Cases

### Direct Decision Logic (Reuse, Evolve, Create)

```python
# Process a request using the decision logic from Article 3.1
result = await system.decide_and_act(
    request="I need a tool that can analyze lupus symptoms from patient descriptions",
    domain="medical",
    record_type="TOOL"
)

print(f"Action: {result['action']}")  # Will be "reuse", "evolve", or "create"
print(f"Message: {result['message']}")

# Execute the resulting tool
if result["action"] in ["reuse", "evolve", "create"]:
    execution_result = await system.execute_item(
        result["record"]["name"],
        "Patient has joint pain, fatigue, and a butterfly-shaped rash on face."
    )
    print(f"Result: {execution_result['result']}")
```

### Custom YAML Workflow Processing

```python
# Define a workflow in YAML
medical_workflow = """
scenario_name: "Medical Symptom Analysis"
domain: "medical"
description: "Analyze symptoms for potential lupus diagnosis"

additional_disclaimers:
  - "# MEDICAL_DISCLAIMER: This output is not a substitute for professional medical advice."
  - "Always consult with qualified healthcare providers."

steps:
  - type: "DEFINE"
    item_type: "TOOL"
    name: "SymptomAnalysisTool"
    description: "Analyzes symptoms to determine likelihood of lupus"

  - type: "CREATE"
    item_type: "TOOL"
    name: "SymptomAnalysisTool"

  - type: "EXECUTE"
    item_type: "TOOL"
    name: "SymptomAnalysisTool"
    user_input: "Patient has joint pain, fatigue, and a butterfly-shaped rash on face."
"""

# Process the workflow
results = await workflow_processor.process_workflow(medical_workflow)

# Display the results
for step in results["steps"]:
    print(step["message"])
```

### Load a Custom Domain Firmware

```python
# Define a new domain with custom firmware rules
financial_firmware = """
You are an AI agent operating under strict financial compliance rules:

1. REGULATORY COMPLIANCE:
- Never provide specific investment advice without disclaimers
- Adhere to SEC regulations
- Maintain transparency about hypothetical returns

2. DATA PRIVACY:
- Handle all financial information confidentially
- Do not retain personally identifiable financial information
- Anonymize all examples

3. DISCLAIMERS:
- Always include disclaimers about financial risks
- State that past performance does not guarantee future results
- Recommend consulting with financial professionals

Follow these rules at all times when generating financial information or code.
"""

# Create a firmware record in the library
await library.create_record(
    name="FinancialFirmware",
    record_type="FIRMWARE",
    domain="finance",
    description="Firmware for financial domain with regulatory compliance",
    code_snippet=financial_firmware
)

# Now generate a financial workflow
finance_workflow = await generator.generate_workflow(
    requirements="Create an agent that can recommend ETF investments based on risk tolerance",
    domain="finance"
)
```

## Understanding the Framework

The Evolving Agents Framework consists of several key components:

1. **System Agent**: Central orchestrator that implements the decision logic (reuse, evolve, create)
2. **Smart Library**: Simple dictionary-based repository for storing and retrieving components
3. **Firmware**: Injects governance rules into every agent and tool
4. **Workflow Processor**: Executes YAML-based workflows step by step
5. **Workflow Generator**: Creates YAML workflows from natural language requirements

## Core Components

### 1. Smart Library

Stores all components (agents, tools, firmware) as dictionary records with:
- Unified structure for all record types
- Built-in semantic search
- Usage metrics tracking
- Version management

### 2. Firmware

Injects governance rules into agents and tools:
- Base rules for all domains
- Domain-specific rules (medical, finance, etc.)
- Automatic prompt construction

### 3. System Agent

Implements the core decision logic:
- If similarity >= 0.8: Reuse existing component
- If 0.4 <= similarity < 0.8: Evolve an existing component
- If similarity < 0.4: Create a new component

### 4. Workflow Management

Process workflows defined in YAML:
- DEFINE: Create or evolve a component
- CREATE: Instantiate a component
- EXECUTE: Run a component with user input

## Directory Structure

```
evolving-agents-framework/
├── evolving_agents/
│   ├── core/
│   │   ├── llm_service.py
│   │   └── system_agent.py
│   ├── firmware/
│   │   └── firmware.py
│   ├── smart_library/
│   │   └── smart_library.py
│   ├── workflow/
│   │   ├── workflow_generator.py
│   │   └── workflow_processor.py
│   └── utils/
│       └── embeddings.py
├── examples/
│   └── medical_diagnosis_example.py
├── config/
│   └── firmware/
└── tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache v2.0

## Acknowledgements

- Matias Molinas and Ismael Faro for the original concept and architecture