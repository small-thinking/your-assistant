import argparse
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.utilities import GoogleSearchAPIWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process input string for AI assistant"
    )
    parser.add_argument(
        "-i", "--input_string", required=True, help="Input string for AI assistant"
    )
    return parser.parse_args()


def test(args: argparse.Namespace):
    google_search_tool = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name="Google Search",
            func=google_search_tool.run,
            description="useful for when you need to ask with search to get time sensitive information",
        )
    ]
    llm = OpenAI(temperature=0)
    agent_chain = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    agent_chain.run(args.input_string)


if __name__ == "__main__":
    args = parse_args()
    test(args=args)
