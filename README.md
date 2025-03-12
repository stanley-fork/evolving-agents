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

For detailed architectural information, see [ARCHITECTURE.md](doc/ARCHITECTURE.md).

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

print(response.result.text)
```

## Use Cases

- **Document Processing**: Create specialized agents for different document types that collaborate to extract and analyze information
- **Healthcare**: Medical agents communicating with pharmacy and insurance agents to coordinate patient care
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

## Final Note:
The code is currently being actively refactored to align with the README.md. Some features described here may not yet be implemented or fully functional. The updated version reflecting this documentation will be available in the next few days. Stay tuned! 🚀