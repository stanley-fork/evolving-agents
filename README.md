# Evolving Agents Framework

A production-grade framework for creating, managing, and evolving AI agents in a controlled environment.

## Features

- **Firmware Injection**: Enforce governance rules across all agents
- **Smart Library**: Centralized repository for agent and tool definitions
- **Semantic Search**: Find and reuse existing agents and tools
- **YAML-based Scenarios**: Configure complex agent workflows with simple YAML files
- **Multi-Agent Orchestration**: Coordinate multiple specialized agents

## Installation

```bash
pip install evolving-agents-framework
```

## Quick Start

```python
from evolving_agents import SystemAgent, Firmware
from evolving_agents.utils import load_yaml_scenario

# Initialize core components
firmware = Firmware()
system_agent = SystemAgent(firmware)

# Load and run a scenario
await system_agent.load_and_process_scenario("examples/medical_diagnosis/medical_scenario.yaml")
```

## Documentation

See the [documentation](docs/getting_started.md) for detailed usage.

## License

APACHE 2.0
