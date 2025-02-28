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
- beeai-framework
- PyYAML
- OpenAI (or another LLM provider of your choice)

## Quick Start

### From Natural Language to Working Agent

```python
import asyncio
import logging

from evolving_agents.smart_library.library_manager import SmartLibrary
from evolving_agents.core.llm_service import LLMService
from evolving_agents.core.system_agent import SystemAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the framework
    library = SmartLibrary("library.json")
    llm = LLMService(provider="openai")  # Requires OpenAI API key in environment
    system = SystemAgent(library, llm)
    
    # Initialize with base firmware for medical domain
    await system.initialize_system("config/system_config.yaml")
    
    # Generate and execute a workflow from natural language requirements
    result = await system.process_request(
        request="Create an agent that can analyze a patient's symptoms, provide a preliminary diagnosis for lupus, and research the latest immunosuppressant treatments. Include medical disclaimers.",
        domain="medical"
    )
    
    # View the results
    print(f"Generated workflow:\n{result['workflow_yaml']}\n")
    print("Execution results:")
    for step in result["execution"]["steps"]:
        print(f"- {step['message']}")
    
    # If the workflow had an EXECUTE step with results, display them
    if any("result" in step for step in result["execution"]["steps"]):
        final_step = [step for step in result["execution"]["steps"] if "result" in step][-1]
        print("\nAgent response:")
        print(final_step["result"])

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
    from_existing_snippet: "MedicalDiagnosisTool"
    evolve_changes:
      docstring_update: "Specialized for lupus diagnosis based on symptoms"
    description: "Analyzes patient symptoms to determine likelihood of lupus"

  - type: "DEFINE"
    item_type: "TOOL"
    name: "ImmunosuppressantResearchTool"
    from_existing_snippet: "MedicalResearchTool"
    evolve_changes:
      docstring_update: "Specialized for immunosuppressant treatment research"
    description: "Researches latest immunosuppressant treatments for lupus"

  - type: "DEFINE"
    item_type: "AGENT"
    name: "LupusDiagnosisAndTreatmentAgent"
    agent_type: "MedicalReActAgent"
    description: "Agent that diagnoses lupus and researches treatments with proper disclaimers"
    required_tools:
      - "LupusDiagnosisTool"
      - "ImmunosuppressantResearchTool"
    disclaimers_in_docstring: true

  - type: "CREATE"
    item_type: "AGENT"
    name: "LupusDiagnosisAndTreatmentAgent"

  - type: "EXECUTE"
    item_type: "AGENT"
    name: "LupusDiagnosisAndTreatmentAgent"
    user_input: "My patient has fatigue, joint pain, skin rash, and fever. What's the likelihood of lupus and what are the latest treatment options?"

Execution results:
- Evolved MedicalDiagnosisTool to LupusDiagnosisTool, injecting firmware for domain 'medical'.
- Evolved MedicalResearchTool to ImmunosuppressantResearchTool, injecting firmware for domain 'medical'.
- Defined new AGENT LupusDiagnosisAndTreatmentAgent
- Created AGENT instance LupusDiagnosisAndTreatmentAgent
- Executed AGENT LupusDiagnosisAndTreatmentAgent

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

1. **Belimumab (Benlysta)** - FDA-approved monoclonal antibody that targets B-lymphocyte stimulator (BLyS)
   - Recent studies show efficacy in reducing disease activity and flares
   - Available in both IV and subcutaneous formulations

2. **Anifrolumab (Saphnelo)** - FDA-approved in 2021
   - Type I interferon receptor antagonist
   - Demonstrated significant reduction in disease activity across multiple organ systems

3. **Voclosporin (Lupkynis)** - Approved January 2021
   - Calcineurin inhibitor specifically for lupus nephritis
   - Shows improved renal response compared to standard therapy

4. **Obinutuzumab** - Currently in Phase III trials
   - Anti-CD20 monoclonal antibody showing promise for lupus nephritis

5. **JAK inhibitors** (Baricitinib, Tofacitinib) - Under investigation
   - Early studies show potential benefit in skin and joint manifestations

Please consult with a rheumatologist to determine the most appropriate treatment based on disease severity, organ involvement, and patient-specific factors.
```

## More Advanced Use Cases

### Smart Selection from the Library

The framework can automatically find and use the most relevant agent from your library:

```python
# Find the best matching agent and execute it directly
result = await system.semantic_find_and_execute(
    query="Need an agent that can diagnose lupus from symptoms",
    input_text="Patient has butterfly rash, joint pain, and fatigue",
    domain="medical"
)

print(f"Selected agent: {result['name']}")
print(f"Result: {result['result']}")
```

### Generate Custom Workflow and Save for Later

```python
# Generate a workflow without executing it
workflow_yaml = await system.workflow_generator.generate_workflow(
    requirements="Create a financial advisor that recommends ETFs based on risk tolerance",
    domain="finance",
    output_path="workflows/etf_advisor.yaml"
)

# Later, execute the saved workflow
results = await system.process_yaml_workflow("workflows/etf_advisor.yaml")
```

### Add a New Domain with Custom Firmware

```python
# Load firmware from a YAML file
await system.firmware_manager.load_firmware_from_yaml("config/firmware/legal_firmware.yaml")

# Now generate a legal agent
result = await system.process_request(
    request="Create an agent that can analyze contracts for common liability issues",
    domain="legal"
)
```

## Understanding the Framework

The Evolving Agents Framework consists of several key components:

1. **System Agent**: Central orchestrator that processes requests and coordinates workflows
2. **Smart Library**: Repository that stores all agents, tools, and firmware with usage metrics
3. **Workflow Generator**: Creates YAML workflows from natural language requirements
4. **Workflow Executor**: Runs the workflow steps, instantiating and executing agents/tools
5. **Firmware Manager**: Loads and manages domain-specific rules injected into all agents and tools

## Advanced Features

- **Semantic Evolution**: Agents and tools evolve based on similarity analysis
- **Usage Metrics**: The library tracks success/failure for each component
- **Validation**: Generated code is validated against firmware requirements
- **Human-in-the-Loop**: Support for expert review of generated agents (via pending status)
- **Embedding-Based Discovery**: Find the most semantically similar components for reuse

## Directory Structure

```
evolving-agents-framework/
├── evolving_agents/
│   ├── agents/             # Agent-related classes
│   ├── core/               # Core orchestration
│   ├── firmware/           # Firmware management
│   ├── smart_library/      # Smart Library components
│   ├── tools/              # Tool-related classes
│   ├── utils/              # Utilities
│   └── workflow/           # Workflow generation and execution
├── examples/               # Example usage scripts
├── config/                 # Configuration files
│   ├── firmware/           # Firmware definition files
│   └── initial_records/    # Initial library records
└── tests/                  # Test cases
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Apache v2.0](LICENSE)

## Acknowledgements

- Matias Molinas and Ismael Faro for the original concept and architecture
- The beeai-framework for inspiration and integration
