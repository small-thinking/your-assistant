"""Run the orchestrator in the command line.
"""
import argparse


# Define the function that initialize the argument parser that has the param of the prompt.
def get_parser():
    parser = argparse.ArgumentParser(description="Orchestrator")
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action="store_true",
        help="Whether to print the verbose output.",
    )
    # Add a choice of the orchestrator, candidate is RevChatGPTOrchestrator only for now.
    parser.add_argument(
        "-o",
        "--orchestrator",
        choices=["RevChatGPTOrchestrator"],
        default="RevChatGPTOrchestrator",
        help="The orchestrator to use.",
    )
    return parser


# Define the function that runs the orchestrator.
def run():
    parser = get_parser()
    args = parser.parse_args()
    # Get the orchestrator from the argument.
    if args.orchestrator == "RevChatGPTOrchestrator":
        from your_assistant.core.orchestrator import RevChatGPTOrchestrator

        orchestrator = RevChatGPTOrchestrator(verbose=args.verbose)
    else:
        raise ValueError(f"Unknown orchestrator: {args.orchestrator}")

    # Loop to wait for command line input
    while True:
        try:
            user_input = input("\nEnter your conversation (exit with ctrl + C): ")
            response = orchestrator.process(prompt=user_input)
            print(response)
        except KeyboardInterrupt:
            # If the user presses Ctrl+C, the program will exit gracefully
            exit(0)


if __name__ == "__main__":
    run()
