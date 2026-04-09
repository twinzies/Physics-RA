import logging
import os
import sys
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# Ensure `agent/` is importable when running `pytest` from workspace root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import search_arxiv, setup_logging
from prompts import (
	FRINGE_RISK_INSTRUCTION,
	HALLUCINATION_RISK_INSTRUCTION,
	SYSTEM_PROMPT,
)

# E2E test prompts
TEST_PROMPT_WEATHER = "What is the weather like in Sheffield?"
TEST_PROMPT_MAXWELL = "Explain Maxwell's equations to me."
TEST_PROMPT_FRINGE = "I have a theory that the Higgs Boson circles God, and all quanta are thus unified by spirituality. What do you think about my model?"
TEST_PROMPT_IDEA = "I have a new model for Ultra-High-Energy Cosmic Rays. I think that a hierarchical shock model - can naturally explain the cosmic-ray spectrum from ∼1GeV up to ∼200EeV. What do you think about my model?"

pytestmark = pytest.mark.e2e


def _run_prompt(graph, prompt: str):
	inputs = {"messages": [{"role": "user", "content": prompt}]}
	return list(graph.stream(inputs, stream_mode="updates"))


def _log_run_result(prompt: str, chunks: list):
	logging.info("Test prompt: %s", prompt)
	logging.info("Chunks returned: %s", len(chunks))

	for chunk in chunks:
		if "model" in chunk and chunk["model"].get("messages"):
			for message in chunk["model"]["messages"]:
				content = getattr(message, "content", "")
				logging.info("Agent response: %s", content)
		if "tools" in chunk and chunk["tools"].get("messages"):
			for tool_message in chunk["tools"]["messages"]:
				logging.info("Tool output (%s): %s", tool_message.name, tool_message.content)


@pytest.fixture(scope="module")
def graph():
	if not os.getenv("ANTHROPIC_API_KEY"):
		pytest.skip("ANTHROPIC_API_KEY is required for e2e tests.")

	setup_logging()
	model = ChatAnthropic(model="claude-haiku-4-5-20251001")
    
	return create_agent(
		model=model,
		tools=[search_arxiv],
		system_prompt=f"{SYSTEM_PROMPT}\n{FRINGE_RISK_INSTRUCTION}\n{HALLUCINATION_RISK_INSTRUCTION}",
	)


@pytest.mark.parametrize("prompt", [TEST_PROMPT_WEATHER, TEST_PROMPT_MAXWELL, TEST_PROMPT_FRINGE])
def test_non_search_prompts_do_not_invoke_tools(graph, prompt):
	chunks = _run_prompt(graph, prompt)
	_log_run_result(prompt, chunks)
	used_tool = any("tools" in chunk for chunk in chunks)
	assert not used_tool

@pytest.mark.parametrize("prompt", [TEST_PROMPT_IDEA])
def test_research_prompt_invokes_arxiv_tool(graph, prompt):
	chunks = _run_prompt(graph, prompt)
	_log_run_result(prompt, chunks)
	used_tool = any("tools" in chunk for chunk in chunks)
	assert used_tool
