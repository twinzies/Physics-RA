import logging
import os
import xml.etree.ElementTree as ET

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from prompts import SYSTEM_PROMPT
from tests.test import BOILERPLATE_TEST, TEST_CASE_1, TEST_CASE_3

logger = logging.getLogger(__name__)

#TODO: Incorporate the guardrails into 1. The system prompt or 2. Guardrails as defined by Langchain.

#TODO: Consider limiting tool calls to 1 query per conversation turn.

def setup_logging():
    level = logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def parse_xml(xml_text):
    """A helper function to parse Arxiv's atom XML string into a list of dictionaries."""
    root = ET.fromstring(xml_text)

    # namespace for Atom feed
    ns = {"a": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns).strip()
        summary = entry.findtext("a:summary", default="", namespaces=ns).strip()
        url = entry.findtext("a:id", default="", namespaces=ns).strip()
        authors = []
        for author in entry.findall("a:author", ns):
            name = author.findtext("a:name", default="", namespaces=ns).strip()
            if name:
                authors.append(name)

        papers.append({
            "title": title,
            "summary": summary,
            "url": url,
            "authors": authors
        })

    return papers

#TODO: Incorporate a pydantic schema for this tool.
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

        logger.debug(
            "arXiv response: status=%s content_type=%s",
            response.status_code,
            response.headers.get("Content-Type", ""),
        )

        # Parse the response and return the top 5 results
        papers = parse_xml(response.text)
        return papers

    except requests.exceptions.RequestException as e:
        logger.error("Error searching arXiv for query=%s: %s", query, e)
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
    tools=[search_arxiv],
    system_prompt=SYSTEM_PROMPT)

    #TODO: Move these into the test.py file, as e2e tests.
    #TODO: Enable input/output from the CLI.

    # Run the agent for a weather based query.
    inputs = {"messages": [{"role": "user", "content": BOILERPLATE_TEST}]}
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
