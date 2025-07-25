# CrewAI Knowledge Chatbot Crew
[![Python](https://img.shields.io/badge/Python-3.10%20--%203.13-blue?logo=python)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency--Managed-orange?logo=python)](https://python-poetry.org/)
[![crewAI](https://img.shields.io/badge/Powered%20by-crewAI-6A5ACD?logo=ai)](https://crewai.com)

Welcome to the CrewaiKnowledgeChatbot Crew project, powered by [CrewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

---

## Installation

Ensure you have Python >=3.10 <=3.13 installed on your system. This project uses [Poetry](https://python-poetry.org/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install Poetry:

```bash
pip install poetry
```

Next, navigate to your project directory and install the dependencies:

1. First lock the dependencies and then install them:
```bash
poetry lock

poetry install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**
> You can obtain an Open AI API key from the [OpenAI developer platform](https://platform.openai.com/docs/overview) 

- Modify `src/crewai_knowledge_chatbot/config/agents.yaml` to define your agents
- Modify `src/crewai_knowledge_chatbot/config/tasks.yaml` to define your tasks
- Modify `src/crewai_knowledge_chatbot/crew.py` to add your own logic, tools and specific args
- Modify `src/crewai_knowledge_chatbot/main.py` to add custom inputs for your agents and tasks

---

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
poetry run crewai_knowledge_chatbot
```

This command initializes the crewai_knowledge_chatbot Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folser

---

## Understanding Your Crew

The crewai_knowledge_chatbot Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the CrewaiKnowledgeChatbot Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Joing our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat wtih our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
