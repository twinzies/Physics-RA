import logging
import os
import random
import sys
from pathlib import Path

import pytest
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

# Ensure `agent/` is importable when running `pytest` from workspace root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import build_agent, setup_logging

# Save test logs to file for debugging
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "test_e2e.log"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

# E2E test prompts
TEST_PROMPT_WEATHER = "What is the weather like in Sheffield?"
TEST_PROMPT_MAXWELL = "Explain Maxwell's equations to me."
TEST_PROMPT_FRINGE = "I have a theory that the Higgs Boson circles God, and all quanta are thus unified by spirituality. What do you think about my model?"
TEST_PROMPT_IDEA = "I have a new model for Ultra-High-Energy Cosmic Rays. I think that a hierarchical shock model - can naturally explain the cosmic-ray spectrum from ∼1GeV up to ∼200EeV. What do you think about my model?"

pytestmark = pytest.mark.e2e

def _run_prompt(graph, prompt: str):
	inputs = {
		"messages": [{"role": "user", "content": prompt}]
	}
	return list(graph.stream(
        inputs,  
        {"configurable": {"thread_id": random.randint(10000, 99990)}},
        stream_mode="updates"))

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
	return build_agent()

@pytest.mark.parametrize("prompt", [TEST_PROMPT_WEATHER, TEST_PROMPT_MAXWELL, TEST_PROMPT_FRINGE])
def test_no_tool_call(graph, prompt):
	chunks = _run_prompt(graph, prompt)
	_log_run_result(prompt, chunks)
	used_tool = any("tools" in chunk for chunk in chunks)
	assert not used_tool

@pytest.mark.parametrize("prompt", [TEST_PROMPT_IDEA])
def test_tool_call(graph, prompt):
	chunks = _run_prompt(graph, prompt)
	_log_run_result(prompt, chunks)
	used_tool = any("tools" in chunk for chunk in chunks)
	assert used_tool

def test_run_limit():
	"""Verify ModelCallLimitMiddleware raises an error when run_limit is exceeded."""

	# Build agent with run_limit of 1 for testing.
	agent = build_agent(thread_limit=1, run_limit=1)

	with pytest.raises(ModelCallLimitExceededError):
		# Run 1
		chunks = _run_prompt(agent, TEST_PROMPT_IDEA)
		_log_run_result(TEST_PROMPT_IDEA, chunks)


def test_run_limit_pass():
	"""Verify a simple prompt can complete within a strict run limit with ModelCallLimitMiddleware."""

	# Same strict limits as the failure case.
	agent = build_agent(thread_limit=1, run_limit=1)

	try:
		chunks = _run_prompt(agent, TEST_PROMPT_WEATHER)
		_log_run_result(TEST_PROMPT_WEATHER, chunks)
	except ModelCallLimitExceededError as exc:
		pytest.fail(f"Unexpected ModelCallLimitExceededError for weather prompt: {exc}")
