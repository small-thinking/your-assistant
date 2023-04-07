"""The orchestrator that uses the agents to orchestrate the conversation.
"""
import argparse
import os
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain.llms.base import LLM
from langchain.memory import ConversationSummaryBufferMemory

from your_assistant.core.indexer import KnowledgeIndexer
from your_assistant.core.llms import ChatGPT, RevBard, RevChatGPT
from your_assistant.core.responder import DocumentQA
from your_assistant.core.utils import Logger, load_env


class Orchestrator(ABC):
    """The abstract orchestrator."""

    def __init__(self, args: argparse.Namespace = argparse.Namespace()):
        load_env()
        self.args = args
        self.verbose = args.verbose
        self.logger = Logger(type(self).__name__)

    @classmethod
    def add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        # Add verbase (default: False) to parser.
        parser.add_argument("-v", "--verbose", default=False, action="store_true")
        cls._add_arguments_to_parser(parser)

    @classmethod
    @abstractmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        """Add the arguments to the parser.

        Args:
            parser (argparse.ArgumentParser): The parser that accepts the arguments.
        """
        pass

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        return cls(args)

    @abstractmethod
    def process(self, args: argparse.Namespace) -> str:
        raise NotImplementedError("process must be implemented.")


class LLMOrchestrator(Orchestrator):
    """The abstract orchestrator that uses the LLM."""

    def __init__(self, args: argparse.Namespace = argparse.Namespace()):
        super().__init__(args)
        self.logger = Logger(type(self).__name__)
        self.llm: Optional[LLM] = None
        self._init_llm(args=args)
        if not self.llm:
            raise ValueError("The llm must be initialized.")
        if args.use_memory:
            self.memory: ConversationSummaryBufferMemory = (
                ConversationSummaryBufferMemory(
                    llm=self.llm, max_token_limit=args.memory_token_size
                )
            )

    def _init_llm(self, args: argparse.Namespace):
        raise NotImplementedError("_init_llm must be implemented.")

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--use-memory",
            action="store_true",
            help="Whether to use the memory.",
        )
        parser.add_argument(
            "--memory-token-size",
            default=300,
            type=int,
            help="The maximum number of tokens used to keep the memory.",
        )

    def process(self, args: argparse.Namespace) -> str:
        # Set memory.
        original_prompt = args.prompt
        if args.use_memory:
            history: Dict[str, Any] = self.memory.load_memory_variables({})
            if self.verbose:
                self.logger.info(f"History: {history}\n\n")
            args.prompt = textwrap.dedent(
                f"""
                Current conversation:
                {history["history"]}
                "User": {args.prompt}
            """
            )
        if self.verbose:
            self.logger.info(f"Prompt: {args.prompt}\n\n")
        response = self._process(args=args)
        if args.use_memory:
            # Only save the user original prompt without history augmentation.
            self.memory.save_context(
                inputs={"user": original_prompt}, outputs={"AI": response}
            )
        return response

    @abstractmethod
    def _process(self, args: argparse.Namespace) -> str:
        raise NotImplementedError("_process must be implemented.")


class ChatGPTOrchestrator(LLMOrchestrator):
    """The orchestrator that uses the ChatGPT."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the orchestrator."""
        super().__init__(args=args)

    def _init_llm(self, args: argparse.Namespace):
        self.model = args.model
        self.temperature = args.temperature
        self.max_token = args.max_token
        self.llm = ChatGPT()

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_arguments_to_parser(parser=parser)
        parser.add_argument("-m", "--model", default="gpt-3.5-turbo", type=str)
        parser.add_argument(
            "--temperature",
            default=0.1,
            type=float,
            help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output"
            " more random, while lower values like 0.2 will make it more focused and deterministic.",
        )
        parser.add_argument(
            "--max-token",
            default=500,
            type=int,
            help="The total length of input tokens and generated tokens is limited by the model's context length.",
        )

    def _process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(args.prompt) == 0:
            return ""
        if not self.llm:
            raise ValueError("The llm must be initialized.")
        response = self.llm(args.prompt)
        return response


class RevChatGPTOrchestrator(LLMOrchestrator):
    """The orchestrator that uses the RevChatGPT."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the orchestrator."""
        super().__init__(args=args)

    def _init_llm(self, args: argparse.Namespace):
        self.llm = RevChatGPT()

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_arguments_to_parser(parser=parser)

    def _process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(args.prompt) == 0:
            return ""
        if not self.llm:
            raise ValueError("The llm must be initialized.")
        response = self.llm(args.prompt)
        return response


class RevBardOrchestrator(LLMOrchestrator):
    """The orchestrator that uses the RevBard."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the orchestrator."""
        super().__init__(args=args)
        args_dict = vars(args)
        for key, value in args_dict.items():
            print(f"{key}: {value}")

    def _init_llm(self, args: argparse.Namespace):
        self.llm = RevBard()

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_arguments_to_parser(parser=parser)

    def _process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            args (argparse.Namespace): The arguments to the orchestrator.
        """
        if len(args.prompt) == 0:
            return ""
        if not self.llm:
            raise ValueError("The llm must be initialized.")
        response = self.llm(args.prompt)
        return response


class KnowledgeIndexOrchestrator(Orchestrator):
    def __init__(self, args: argparse.Namespace):
        """Initialize the orchestrator."""
        super().__init__(args=args)
        self.indexer = KnowledgeIndexer(db_name=args.db_name, verbose=args.verbose)

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_arguments_to_parser(parser=parser)
        parser.add_argument(
            "-d",
            "--db_name",
            default="faiss.db",
            type=str,
            help="The name of the database to store the embeddings. Default is faiss.db.",
        )
        parser.add_argument(
            "-p",
            "--path",
            required=True,
            type=str,
            help="The path to the data to be indexed. It can be a file path or a directory path.",
        )
        parser.add_argument(
            "-c",
            "--chunk_size",
            default=500,
            type=int,
            help="The size of the chunk to partition the document into sections for embedding. Default is 500.",
        )
        parser.add_argument(
            "-o",
            "--chunk_overlap",
            default=50,
            type=int,
            help="The overlap of the chunk to partition the document into sections for embedding. Default is 50.",
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        return cls(args=args)

    def process(self, args: argparse.Namespace) -> str:
        """Process the index according to the path.

        Args:
            args (argparse.Namespace): The arguments to the orchestrator.
        """
        path, chunk_size, chunk_overlap = args.path, args.chunk_size, args.chunk_overlap
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")
        # Use a for loop to index each file if path is a directory.
        # Use an array to store the response of each index, and then concatenate them.
        if os.path.isdir(path):
            if self.verbose:
                self.logger.info(f"Indexing files in {path}...")
            responses = []
            for file in os.listdir(path):
                if file.startswith("."):
                    continue
                file_path = os.path.join(path, file)
                response = self.indexer.index(
                    path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                responses.append(response)
            return "\n".join(responses)
        else:
            response = self.indexer.index(
                path=path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            return response


class QAOrchestrator(LLMOrchestrator):
    """The orchestrator that uses the QA agent."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the orchestrator."""
        super().__init__(args=args)
        self.qa = DocumentQA(
            db_name=args.db_name,
            llm_type=args.llm_type,
            test_mode=args.test_mode,
            verbose=args.verbose,
        )

    def _init_llm(self, args: argparse.Namespace):
        if args.llm_type == "ChatGPT":
            self.llm = ChatGPT()

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_arguments_to_parser(parser=parser)
        parser.add_argument(
            "-d",
            "--db_name",
            default="faiss.db",
            type=str,
            help="The name of the database to store the embeddings. Default: faiss.db.",
        )
        parser.add_argument(
            "-l",
            "--llm_type",
            default="ChatGPT",
            type=str,
            help="The type of the language model to use. Default: ChatGPT.",
        )
        parser.add_argument(
            "-t",
            "--test_mode",
            default=False,
            action="store_true",
            help="Test the model. Default: False.",
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        return cls(args=args)

    def _process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(args.prompt) == 0:
            return ""
        if args.verbose:
            self.logger.info(f"Prompt: {args.prompt}")
        response = self.qa.answer(args.prompt)
        return response
