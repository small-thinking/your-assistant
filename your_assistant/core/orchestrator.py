"""The orchestrator that uses the agents to orchestrate the conversation.
"""
import argparse
import os
from abc import ABC, abstractmethod

from your_assistant.core.indexer import KnowledgeIndexer
from your_assistant.core.llms import RevBard, RevChatGPT
from your_assistant.core.responder import DocumentQA
from your_assistant.core.utils import Logger, load_env


class Orchestrator(ABC):
    """The abstract orchestrator."""

    def __init__(self, verbose: bool = False):
        load_env()
        self.verbose = verbose
        self.logger = Logger(type(self).__name__)

    @classmethod
    def add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        # Add verbase (default: False) to parser.
        parser.add_argument("-v", "--verbose", default=False, action="store_true")
        cls._add_arguments_to_parser(parser)

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError(
            f"Class {cls.__name__} must implement _add_arguments_to_parser."
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        raise NotImplementedError(
            f"Class {cls.__name__} must implement create_from_args."
        )

    @abstractmethod
    def process(self, args: argparse.Namespace) -> str:
        raise NotImplementedError("process must be implemented.")


class RevChatGPTOrchestrator(Orchestrator):
    """The orchestrator that uses the RevChatGPT."""

    def __init__(self, verbose: bool = False):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.llm = RevChatGPT()

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        """Create the orchestrator from the arguments."""
        return cls(verbose=args.verbose)

    def process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(args.prompt) == 0:
            return ""
        response = self.llm(args.prompt)
        return response


class RevBardOrchestrator(Orchestrator):
    """The orchestrator that uses the RevBard."""

    def __init__(self, verbose: bool = False):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.llm = RevBard()

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        pass

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        """Create the orchestrator from the arguments."""
        return cls(verbose=args.verbose)

    def process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            args (argparse.Namespace): The arguments to the orchestrator.
        """
        if len(args.prompt) == 0:
            return ""
        response = self.llm(args.prompt)
        return response


class KnowledgeIndexOrchestrator(Orchestrator):
    def __init__(self, db_name: str = "faiss.db", verbose: bool = False):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.indexer = KnowledgeIndexer(db_name=db_name, verbose=verbose)

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-d",
            "--db_name",
            default="faiss.db",
            type=str,
            help="The name of the database to store the embeddings.",
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
            default=1000,
            type=int,
            help="The size of the chunk to partition the document into sections for embedding.",
        )
        parser.add_argument(
            "-o",
            "--chunk_overlap",
            default=100,
            type=int,
            help="The overlap of the chunk to partition the document into sections for embedding.",
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        return cls(
            db_name=args.db_name,
            verbose=args.verbose,
        )

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


class QAOrchestrator(Orchestrator):
    """The orchestrator that uses the QA agent."""

    def __init__(
        self,
        db_name: str = "faiss.db",
        llm_type: str = "RevChatGPT",
        test_mode: bool = False,
        verbose: bool = False,
    ):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.qa = DocumentQA(db_name=db_name, llm_type=llm_type, test_mode=test_mode)

    @classmethod
    def add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        # Add verbase (default: False) to parser.
        parser.add_argument("-v", "--verbose", default=False, action="store_true")
        cls._add_arguments_to_parser(parser)

    @classmethod
    def _add_arguments_to_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-d",
            "--db_name",
            default="faiss.db",
            type=str,
            help="The name of the database to store the embeddings.",
        )
        parser.add_argument(
            "-l",
            "--llm_type",
            default="RevChatGPT",
            type=str,
            help="The type of the language model to use.",
        )
        parser.add_argument(
            "-t",
            "--test_mode",
            default=False,
            action="store_true",
            help="Test the model.",
        )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace) -> "Orchestrator":
        return cls(
            db_name=args.db_name,
            llm_type=args.llm_type,
            test_mode=args.test_mode,
            verbose=args.verbose,
        )

    def process(self, args: argparse.Namespace) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(args.prompt) == 0:
            return ""
        response = self.qa.answer(args.prompt)
        return response
