import argparse
import logging
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from prompts import (
    FRINGE_RISK_INSTRUCTION,
    HALLUCINATION_RISK_INSTRUCTION,
    SYSTEM_PROMPT,
)
from pydantic import BaseModel, Field

MODEL = "claude-haiku-4-5-20251001"

# Note, these can be written as fraction of context window size, however harder to test for given project.
SUMMARIZATION_TRIGGER = 6000 
SUMMARIZATION_KEEP = 3000

logger = logging.getLogger(__name__)

# Load environment variables automatically
load_dotenv()

def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
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

import asyncio

######## Concurrency conversion ########
import httpx


@tool(args_schema=ArxivSearchInput)
async def search_arxiv_async(query: str, max_results: int = 5)->list:
    """Searches arXiv for a given search query with asynchronous concurrency and returns a list of the top results.
    """
    logger.info("Async tool search_arxiv called with query=%s", query)

    try:
        url = "http://export.arxiv.org/api/query?"
        params = {
            "search_query": query,
            "start":0,
            "max_results": max_results
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()

        logger.debug(
            "arXiv response: status=%s content_type=%s",
            response.status_code,
            response.headers.get("Content-Type", ""),
        )

        # Parse the response and return the top 5 results
        papers = parse_xml(response.text)
        return papers
    
    except httpx.RequestError as e:
        logger.error("Error searching arXiv for query=%s: %s", query, e)
        return f"Error searching arXiv: {e}"

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

async def chat(message, agent, thread_id):
    inputs = {"messages": [{"role": "user", "content": message}]}
    final_response = ""
    print("Agent: ", end="", flush=True)

#### For loops will need to be converted to async for loops ####
    async for chunk in agent.astream(
        inputs,  
        {"configurable": {"thread_id": thread_id}},
        stream_mode="updates"):
        if "model" not in chunk:
            continue

        for model_message in chunk["model"].get("messages", []):
            # Skip intermediary assistant messages that are paired with tool calls.
            if getattr(model_message, "tool_calls", None):
                continue

            content = getattr(model_message, "content", "")
            if isinstance(content, str) and content:
                final_response = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    final_response = "\n".join(text_parts).strip()

    if not final_response:
        final_response = "No response generated."
    print(final_response)
    return {"message": final_response}

def build_agent(thread_limit=15, run_limit=5):
    """Build and return the configured agent graph."""
    checkpointer = InMemorySaver()
    
    middleware = [
        SummarizationMiddleware(
            model=MODEL,
            trigger=("tokens", SUMMARIZATION_TRIGGER),
            keep=("tokens", SUMMARIZATION_KEEP),
        ),
        ModelCallLimitMiddleware(
            thread_limit=thread_limit,
            run_limit=run_limit,
            exit_behavior="error",
        ),
    ]

    # Guardrails included within the system prompt.
    return create_agent(
        model=MODEL,
        tools=[search_arxiv_async],
        system_prompt=f'{SYSTEM_PROMPT}\n{FRINGE_RISK_INSTRUCTION}\n{HALLUCINATION_RISK_INSTRUCTION}',
        checkpointer=checkpointer,
        middleware=middleware,
    )

def main():
    parser = argparse.ArgumentParser(description="Physics research assistant")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logs",
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    graph = build_agent()

    logger.info("Agent initialized with model: %s", MODEL)

    thread_id = 100
    print(f"Starting a new conversation thread (thread {thread_id}).")
    
    # Greet the user
    print("Hi there, I'm your physics expert assistant.")
    print("I'm here to help you explore new concepts and discuss novel ideas.")
    print("Type /new to start a new thread, /exit or /quit to stop.")

    # Chat loop
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

        if user_message.lower() in {"/new"}:
            thread_id += 1
            print(f"Starting a new conversation thread (thread {thread_id}).")
            continue

        asyncio.run(chat(user_message, graph, thread_id=thread_id))

if __name__ == "__main__":
    main()
