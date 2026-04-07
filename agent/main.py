import logging
import os

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from prompts import SYSTEM_PROMPT
from tests.test import BOILER_TEST, TEST_CASE_1, TEST_CASE_3

logger = logging.getLogger(__name__)


def setup_logging():
    level = logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

@tool
def get_weather(location: str):
    """Returns the weather in a given location.

    Args:
        location: The location for which weather information is required.
    """
    logger.info("Tool get_weather called with location=%s", location)
    return f"The weather in {location} is cloudy with a chance of meatballs."

@tool
def search_arxiv(query: str):
    """Searches arXiv for a given search query and returns the top 5 results.

    Args:
        query(str): The keyword or topic string to search for on Arxiv.
    """
    logger.info("Tool search_arxiv called with query=%s", query)

    try:
        url = "http://export.arxiv.org/api/query?"
        params = {
            "search_query": query,
            "start":0,
            "max_results": 5}

        # Make the request to arXiv with a timeout of 10s.
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        logging.debug("Response from arXiv: %s", response.text)
        # Parse the response and return the top 5 results
        return response.text

    except requests.exceptions.RequestException as e:
        logger.error("Error searching arXiv: %s", e)
        return f"Error searching arXiv: {e}"
    

# @tool
# def get_arxiv_paper(location: str):

def main():
    setup_logging()

    # Configure the model
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")

    # Create the agent
    graph = create_agent(
    model=model,
    tools=[get_weather, search_arxiv],
    system_prompt=SYSTEM_PROMPT)

    # Run the agent for weather tool.
    inputs = {"messages": [{"role": "user", "content": BOILER_TEST}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)

    # Run the agent for a single query that should NOT invoke any tool.
    inputs = {"messages": [{"role": "user", "content": TEST_CASE_1}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)

    # Run the agent for a single query that should invoke the arxiv tool.
    inputs = {"messages": [{"role": "user", "content": TEST_CASE_3}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)

if __name__ == "__main__":
    main()
