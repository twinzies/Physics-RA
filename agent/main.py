from langchain.agents import create_agent
from prompts import SYSTEM_PROMPT
from tests.test import TEST_CASE_1, BOILER_TEST
import logging
from langchain_anthropic import ChatAnthropic


def get_weather(location: str):
    return f"The weather in {location} is cloudy with a chance of meatballs."

def main():

    # Configure the model
    model = ChatAnthropic("claude-haiku-4-5-20251001")

    # Create the agent
    graph = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT)

    # Run the agent for a single query
    inputs = {"messages": [{"role": "user", "content": BOILER_TEST}]}
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)
    

if __name__ == "__main__":
    main()
