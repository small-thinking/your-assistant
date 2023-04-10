"""Run the orchestrator in the command line.
"""
from colorama import Fore, Style

import your_assistant.core.utils as utils
from your_assistant.core.orchestrator import *

ORCHESTRATORS = {
    "ChatGPT": ChatGPTOrchestrator,
    "Claude": AnthropicOrchestrator,
    "RevChatGPT": RevChatGPTOrchestrator,
    "RevBard": RevBardOrchestrator,
    "QA": QAOrchestrator,
    "KnowledgeIndex": KnowledgeIndexOrchestrator,
}


# Define the function that runs the orchestrator.
def run():
    parser = utils.init_parser(ORCHESTRATORS)
    args = parser.parse_args()
    orchestrator_cls = ORCHESTRATORS[args.orchestrator]
    orchestrator = orchestrator_cls.create_from_args(args)
    params = vars(args)
    print(f"You are using {args.orchestrator}, with parameters: {params}")
    # Init path as user_input if is KnowledgeIndexOrchestrator.
    if args.orchestrator == "KnowledgeIndex":
        response = orchestrator.process(args)
        print(response)
    elif args.orchestrator in [
        "ChatGPT",
        "Claude",
        "RevChatGPT",
        "RevBard",
        "QA",
    ]:
        # Init prompt as user_input if is RevChatGPTOrchestrator, RevBardOrchestrator, QAOrchestrator.
        while True:
            try:
                user_input = input(
                    Fore.GREEN
                    + "\nEnter your conversation (exit with ctrl + C): "
                    + Style.RESET_ALL
                )
                args.prompt = user_input
                response = orchestrator.process(args)
                print(Fore.BLUE + response + Style.RESET_ALL)
            except KeyboardInterrupt:
                exit(0)
    else:
        raise ValueError("The orchestrator is not supported.")


if __name__ == "__main__":
    run()
