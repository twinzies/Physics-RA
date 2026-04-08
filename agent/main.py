import logging
import os
import xml.etree.ElementTree as ET

import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from prompts import (
    FRINGE_RISK_INSTRUCTION,
    HALLUCINATION_RISK_INSTRUCTION,
    SYSTEM_PROMPT,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

def setup_logging():
    level = logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def parse_xml(xml_text)->list:
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

class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="The keyword or topic string to search for on Arxiv.")
    max_results: int = Field(default=5, description="The maximum number of results to return.")

@tool(args_schema=ArxivSearchInput)
def search_arxiv(query: str, max_results: int = 5)->list:
    """Searches arXiv for a given search query and returns a list of the top results.
    """
    logger.info("Tool search_arxiv called with query=%s", query)

    try:
        url = "http://export.arxiv.org/api/query?"
        params = {
            "search_query": query,
            "start":0,
            "max_results": max_results}

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

# TODO: Implement a summarizer middleware for the tool below - useful with memory: https://docs.langchain.com/oss/python/langchain/short-term-memory 

# TODO: Time permitting, add a tool to get a specific paper from Arxiv.
# @tool
# def get_arxiv_paper(location: str):

def chat(message, agent):
    inputs = {"messages": [{"role": "user", "content": message}]}
    final_response = ""
    print("Agent: ", end="", flush=True)

    for model_chunk, _ in agent.stream(inputs, stream_mode="messages"):
        content = getattr(model_chunk, "content", "")
        if isinstance(content, str) and content:
            print(content, end="", flush=True)
            final_response += content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        print(text, end="", flush=True)
                        final_response += text

    if not final_response:
        final_response = "No response generated."
        print(final_response, end="", flush=True)

    print()
    return {"message": final_response}

def main():
    setup_logging()

    # Configure the model
    model = ChatAnthropic(model="claude-haiku-4-5-20251001")

    # Create the agent
    graph = create_agent(
    model=model,
    tools=[search_arxiv],
    system_prompt=f'{SYSTEM_PROMPT}\n{FRINGE_RISK_INSTRUCTION}\n{HALLUCINATION_RISK_INSTRUCTION}') # Guardrails included within the system prompt.

    logger.info("Agent initialized with model: %s", model.model)

    print("Hi there, I'm your physics research assistant.")
    print("I'm here to help you explore new concepts and discuss novel ideas.")
    print("Type /exit or /quit to stop.")

    # TODO: Consider adding a timeout.
    while True:
        try:
            user_message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_message:
            continue

        if user_message.lower() in {"exit", "quit", "/exit", "/quit"}:
            print("Goodbye")
            break

        chat(user_message, graph)


if __name__ == "__main__":
    main()
