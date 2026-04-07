import logging

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from prompts import SYSTEM_PROMPT
from tests.test import BOILER_TEST, TEST_CASE_1

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def get_weather(location: str):
    """A mock tool to get the weather in a location."""
    logger.info("Tool get_weather called with location=%s", location)
    return f"The weather in {location} is cloudy with a chance of meatballs."

def main():
    setup_logging()

    # Configure the model
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")

    # Create the agent
    graph = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT)

    # Run the agent for a single query
    inputs = {"messages": [{"role": "user", "content": BOILER_TEST}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)

    # Run the agent for a single query - should NOT invoke tool.
    inputs = {"messages": [{"role": "user", "content": TEST_CASE_1}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)    

if __name__ == "__main__":
    main()
