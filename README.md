# Evolving Agents Toolkit

A toolkit for agent autonomy, evolution, and governance. Create agents that can understand requirements, evolve through experience, communicate effectively, and build new agents and tools - all while operating within governance guardrails.

![Evolving Agents](evolving-agents-logo.png)

## Why the World Needs This Toolkit

Current agent systems are designed primarily for humans to build and control AI agents. The Evolving Agents Toolkit takes a fundamentally different approach: agents building agents.

![The Evolving Agents Toolkit](1-toolkit.png)

Our toolkit provides:

- **Autonomous Evolution**: Agents learn from experience and improve themselves without human intervention
- **Agent Self-Discovery**: Agents discover and collaborate with other specialized agents to solve complex problems
- **Governance Firmware**: Enforceable guardrails that ensure agents evolve and operate within safe boundaries
- **Self-Building Systems**: The ability for agents to create new tools and agents when existing ones are insufficient
- **Agent-Centric Architecture**: Communication and capabilities built for agents themselves, not just their human creators

Instead of creating yet another agent framework, we build on existing frameworks like BeeAI and OpenAI Agents SDK to create a layer that enables agent autonomy, evolution, and self-governance - moving us closer to truly autonomous AI systems that improve themselves while staying within safe boundaries.

## Architect-Zero: Our Flagship Meta-Agent

Our toolkit is best demonstrated through Architect-Zero, an agent that autonomously designs solutions to complex problems, leveraging LLM intelligence to find the optimal components for tasks.

```python
# Create an Architect-Zero agent
architect_agent = await create_architect_zero(
    llm_service=llm_service,
    smart_library=smart_library,
    agent_bus=agent_bus,
    system_agent_factory=SystemAgentFactory.create_agent
)

# Give it a task to design a solution
task_requirement = """
Create a comprehensive medical diagnostic system that analyzes patient medical records to provide clinical insights. The system should:

1. Extract and structure patient information, vital signs, medications, and symptoms
2. Analyze symptoms to identify possible medical conditions and their likelihood
3. Check for potential interactions between current medications and suggest adjustments
4. Generate treatment recommendations based on the identified conditions
5. Provide appropriate medical disclaimers and highlight when specialist consultation is needed

The system should leverage existing components, evolve them where needed, and create new ones for missing functionality.
"""

# Architect-Zero analyzes the requirements and designs a solution
result = await architect_agent.run(task_requirement)
```

### What Happens Behind the Scenes

Architect-Zero demonstrates the full capabilities of our toolkit:

1. **LLM-Enhanced Analysis**: It intelligently extracts required capabilities from the task requirements
   ```
   Extracted capabilities: ['medical_record_analysis', 'symptom_analysis', 'medication_interaction_check', 'treatment_recommendation', 'medical_disclaimer_generation']
   ```

2. **Smart Component Discovery**: It searches for components that match these capabilities using LLM-powered semantic matching
   ```
   Found component for capability medical_record_analysis using LLM matching: MedicalRecordAnalyzer
   ```

3. **Capability-Based Design**: It designs a complete workflow with specialized components:
   ```yaml
   scenario_name: Medical Diagnostic Workflow
   domain: healthcare
   description: >
     This workflow processes patient medical records to extract information, analyze symptoms,
     identify possible conditions, check medication interactions, and recommend treatments
     with appropriate medical disclaimers.
   
   steps:
     - type: EXECUTE
       item_type: AGENT
       name: MedicalRecordAnalyzer
       inputs:
         medical_record: |
           PATIENT MEDICAL RECORD
           # Full medical record here
       outputs:
         - structured_patient_data
    
     # Additional steps for symptom analysis, medication interactions, etc.
   ```

4. **Component Evolution and Creation**: It determines when to evolve existing components or create new ones:
   ```
   - type: DEFINE
     item_type: AGENT
     name: TreatmentRecommender
     code_snippet: |
       # Implementation code
   ```

5. **Workflow Execution**: The system executes this workflow, processing medical records through all components:
   ```
   === MEDICAL ANALYSIS ===
   Primary condition: Migraine with aura
   Confidence: High
   
   Supporting evidence:
   - Visual symptoms (flashing lights preceding headaches)
   - Nausea accompanying headaches
   - Blurred vision with headaches
   - Partial response to screen time reduction
   
   Medication concerns:
   - Need for both preventive and acute migraine treatment
   - Ensure new migraine medication doesn't interact with existing medications
   ```

6. **System Evolution**: The system can evolve its analysis as new information becomes available:
   ```
   === Evolution of Analysis ===
   New symptoms identified:
   - Visual aura (flashing lights)
   - Blurred vision with headaches
   - Nausea with severe headaches
   
   Key insights from evolution:
   - System recognized pattern shift from tension/hypertension to migraine
   - Incorporated family history that became more relevant with new symptoms
   ```

This example showcases the true potential of our toolkit - a meta-agent that can design, implement, and orchestrate complex multi-agent systems based on high-level requirements, with the ability to evolve as new information becomes available.

## Industry-Specific Examples

### Finance: Invoice Processing System

Our `architect_zero_financial_demo.py` example demonstrates how Architect-Zero designs a multi-agent invoice processing system:

```python
# Give Architect-Zero a financial task
task_requirement = """
Create an advanced invoice processing system that improves upon the basic version. The system should:

1. Use a more sophisticated document analyzer to detect invoices with higher confidence
2. Extract comprehensive information (invoice number, date, vendor, items, subtotal, tax, total)
3. Verify calculations to ensure subtotal + tax = total
4. Generate a structured summary with key insights
5. Handle different invoice formats and detect potential errors
"""

# Architect-Zero designs a financial solution
result = await architect_agent.run(task_requirement)
```

### Healthcare: Medical Diagnostic System

Our `architect_zero_medical_demo.py` example shows how Architect-Zero creates a medical diagnostic system:

```python
# Give Architect-Zero a healthcare task
task_requirement = """
Create a comprehensive medical diagnostic system that analyzes patient medical records. The system should:

1. Extract and structure patient information, vital signs, medications, and symptoms
2. Analyze symptoms to identify possible medical conditions and their likelihood
3. Check for potential interactions between current medications
4. Generate treatment recommendations based on the identified conditions
5. Provide appropriate medical disclaimers for all recommendations
"""

# Architect-Zero designs a healthcare solution
result = await architect_agent.run(task_requirement)
```

## Why is Firmware Essential in Autonomous Agent Evolution?

In a system where agents and tools can evolve autonomously and create new components from scratch, governance firmware becomes not just important but essential. Without proper guardrails:

![Governance Firmware](2-firmware.png)

- **Capability Drift**: Evolved agents could develop capabilities that stray from their intended purpose
- **Alignment Challenges**: Self-improving systems may optimize for the wrong objectives without proper constraints
- **Safety Concerns**: Autonomous creation of new agents could introduce unforeseen risks or harmful behaviors
- **Compliance Issues**: Evolved agents might unknowingly violate regulatory requirements or ethical boundaries

Our firmware system addresses these challenges by embedding governance rules directly into the evolution process itself. It ensures that:

1. All evolved agents maintain alignment with human values and intentions
2. Component creation and evolution happens within clearly defined ethical and operational boundaries
3. Domain-specific compliance requirements (medical, financial, etc.) are preserved across generations
4. Evolution optimizes for both performance and responsible behavior

The firmware acts as a constitution for our agent ecosystem - allowing freedom and innovation within sensible boundaries.

## LLM-Enhanced Smart Library

The Smart Library serves as the institutional memory and knowledge base for our agent ecosystem, now enhanced with LLM capabilities for intelligent component selection:

![Smart Library](3-smartlibrary.png)

- **LLM-Powered Component Selection**: Uses advanced language models to match capabilities with the best components
- **Semantic Component Discovery**: Finds components based on capability understanding rather than exact matches
- **Capability-Based Search**: Understands what a component can do rather than just matching keywords
- **Performance History Integration**: Tracks component success rates to improve selection over time
- **Experience-Based Evolution**: Uses past performance to guide improvements in component capabilities

By using LLMs to understand requirements and match them to capabilities, the Smart Library enables a more intelligent reuse of components, significantly accelerating development of agent-based systems.

## Why Do We Need Agent and Tool Evolution?

Evolution capabilities are essential because no agent or tool is perfect from the start. Evolution enables:

![Agent and Tool Evolution](5-evolution.png)

- **Performance Improvement**: Refining agents based on observed successes and failures
- **Adaptation to Change**: Updating tools when external services or requirements change
- **Specialization**: Creating domain-specific variants optimized for particular use cases
- **Knowledge Transfer**: Applying learnings from one domain to another through targeted adaptation
- **Interface Alignment**: Adjusting agents to work better with new LLMs or companion tools

Evolution represents the core learning mechanism of our system, allowing it to improve over time through experience rather than requiring constant human intervention and rebuilding.

## Why Create Agents and Tools from Scratch?

While evolution is powerful, sometimes entirely new capabilities are needed. Creation from scratch:

![Creating New Agents and Tools](6-new.png)

- **Fills Capability Gaps**: Creates missing components when no suitable starting point exists
- **Implements Novel Approaches**: Builds components that use fundamentally new techniques
- **Introduces Diversity**: Prevents the system from getting stuck in local optima by introducing fresh approaches
- **Responds to New Requirements**: Addresses emerging needs that weren't anticipated in existing components
- **Leverages LLM Strengths**: Utilizes the code generation capabilities of modern LLMs to create well-designed components

The creation capability ensures that our system can expand to meet new challenges rather than being limited to its initial design, making it truly adaptable to changing needs.

## The Agent Bus: Capability-Based Communication

The Agent Bus facilitates communication between agents based on capabilities rather than identity, enabling:

- **Dynamic Discovery**: Agents find each other based on what they can do, not who they are
- **Loose Coupling**: Components can be replaced or upgraded without disrupting the system
- **Resilient Architecture**: The system can continue functioning even when specific agents change
- **Emergent Collaboration**: New collaboration patterns can form without explicit programming

In our medical example, the components registered their capabilities with the Agent Bus, allowing the system to find the right component for each diagnostic stage automatically.

## Key Features

- **Intelligent Agent Evolution**: Tools encapsulate the logic to determine when to reuse, evolve, or create new components
- **Agent-to-Agent Communication**: Agents communicate through capabilities rather than direct references
- **LLM-Enhanced Smart Library**: Find relevant components using advanced LLM understanding of requirements
- **Multi-Strategy Evolution**: Multiple evolution strategies (standard, conservative, aggressive, domain adaptation)
- **Human-readable YAML Workflows**: Define complex agent collaborations with simple, version-controlled YAML
- **Multi-Framework Support**: Seamlessly integrate agents from different frameworks (BeeAI, OpenAI Agents SDK, etc.)
- **Governance through Firmware**: Enforce domain-specific rules across all agent types
- **Agent Bus Architecture**: Connect agents through a unified communication system with pluggable backends
- **Meta-Agents**: Agents like Architect-Zero that can design and create entire agent systems

For detailed architectural information, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Supported Frameworks

### BeeAI Framework
Our core agent architecture is built on BeeAI's ReActAgent system, providing reasoning-based decision making.

### OpenAI Agents SDK
We fully support the OpenAI Agents SDK, enabling:
- Creation and execution of OpenAI agents within our system
- Experience-based evolution of OpenAI agents
- Firmware rules translated to OpenAI guardrails
- A/B testing between original and evolved agents
- YAML workflow integration across frameworks

## Quick Start

```bash
# Clone the repository
git clone https://github.com/matiasmolinas/evolving-agents.git
cd evolving-agents

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install OpenAI Agents SDK
pip install -r requirements-openai-agents.txt

# Run the Finance example
python examples/architect_zero_financial_demo.py

# Run the Healthcare example
python examples/architect_zero_medical_demo.py
```

## System Initialization Example

```python
# Initialize core components
llm_service = LLMService(provider="openai", model="gpt-4o")
smart_library = SmartLibrary("smart_library.json", llm_service)
agent_bus = SimpleAgentBus("agent_bus.json")

# Create the system agent
system_agent = await SystemAgentFactory.create_agent(
    llm_service=llm_service,
    smart_library=smart_library,
    agent_bus=agent_bus
)

# Create the Architect-Zero agent
architect_agent = await create_architect_zero(
    llm_service=llm_service,
    smart_library=smart_library,
    agent_bus=agent_bus,
    system_agent=system_agent
)

# Now you can use architect_agent.run() to solve complex problems
```

## Key Technical Achievements

1. **LLM-Enhanced Smart Library**: Uses language models to intelligently match capabilities to components
2. **Agent-Design-Agent**: Architect-Zero can design and implement entire agent systems
3. **Tool-Encapsulated Logic**: Each tool contains its own strategy, enabling independent evolution
4. **Pure ReActAgent Implementation**: All agents use reasoning rather than hardcoded functions
5. **Cross-Framework Integration**: Seamless interaction between BeeAI and OpenAI agents
6. **Experience-Based Evolution**: Agents evolve based on performance metrics and usage patterns
7. **Unified Governance**: Firmware rules apply to all agent types through appropriate mechanisms

## Use Cases

- **Document Processing**: Create specialized agents for different document types that collaborate to extract and analyze information
- **Healthcare**: Medical diagnostic agents communicating with pharmacy and insurance agents to coordinate patient care
- **Financial Analysis**: Portfolio management agents collaborating with market analysis agents
- **Customer Service**: Routing agents delegating to specialized support agents
- **Multi-step Reasoning**: Break complex problems into components handled by specialized agents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Apache v2.0](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=matiasmolinas/evolving-agents&type=Timeline)](https://star-history.com/#matiasmolinas/evolving-agents&Timeline)

## Acknowledgements

- [Matias Molinas](https://github.com/matiasmolinas) and [Ismael Faro](https://github.com/ismaelfaro) for the original concept and architecture
- BeeAI framework for integrated agent capabilities
- OpenAI for the Agents SDK