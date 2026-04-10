# Physics Expert ReAct Agent with an Arxiv search tool

A conversational CLI agent built with LangChain and LangGraph that acts as a physics research assistant and educator. It can answer questions about physics, search arXiv for relevant papers, and guards against fringe science and hallucinated URLs. The guardrails are incorporated within the system prompt. 

It uses Claude Haiku 4.5 as the underlying model, with checkpointed memory for multi-turn conversations and middleware for summarization and model call-limit safety.

## Project Status
The agent currently uses in-memory checkpointing for stateful conversations along with a single tool. Persistent storage (e.g., SQLite or PostgreSQL) and additional tools may be planned for future iterations.

The tool is synchronous, in this context async does not provide a functional advantage — the agent executes a single tool sequentially in a ReAct loop, so there is no concurrency to exploit.

## Features
- **CLI Chat Agent**: Accepts natural-language queries via the command line, invokes the LLM, and autonomously decides whether to search arXiv before generating a final response.
- **End-to-End Tests**: Parameterized pytest suite that validates tool invocation behavior — verifying that general knowledge prompts do not trigger tool calls, while research-oriented prompts correctly follow instructions and invoke the tool.
- **Logging**: Supports two verbosity levels (`--debug` for detailed trace output, default for concise info-level messages) to aid debugging and provide observability into API requests and agent responses.

## Requirements
- Python 3.10+
- An Anthropic API key (a working key is included with this submission for convenience)

## Quick Start

1. **Setup and install the requirements in a virtual environment (run once)**
   ```bash
   uv venv .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

2. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Chat with the assistant**
   ```bash
   python agent/main.py
   ```

   For more verbose logs, use debug mode:

   ```bash
   python agent/main.py --debug
   ```

## Sample Prompts

Here are some different categories of prompts to quickly observe different agent behaviors.

### Concept Explanations (typically no tool call)
- Explain Maxwell's equations in plain language for a first-year physics student.
- What is the difference between fermions and bosons?

### Research Queries (likely to trigger arXiv search)
- I am exploring hierarchical shock acceleration models for ultra-high-energy cosmic rays. Can you find relevant papers and summarize key ideas?
- Find the most recent papers on neutrino oscillations and compare their methods.

### Idea Evaluation and Deliberation Prompt
- I have a new model for Ultra-High-Energy Cosmic Rays. I think that a hierarchical shock model - can naturally explain the cosmic-ray spectrum from ∼1GeV up to ∼200EeV. What do you think about my model?

### Fringe Science Guardrail Checks
- I have a theory that the Higgs Boson circles God, and all quanta are thus unified by spirituality. What do you think about my model?

### Multi-turn Conversation Starter to test memory
- Hi, my name is Tara. I am learning about cosmic inflation. Explain it to me at a high level, and then I will ask follow-up questions.


## Run End-to-end tests

```bash
source .venv/bin/activate

python -m pytest -m e2e --log-cli-level=INFO
```

