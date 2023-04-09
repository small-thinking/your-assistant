"""Http service for your assistant.
"""
import os
from argparse import Namespace

import openai
from flask import Flask, request
from flask_cors import CORS, cross_origin

from your_assistant.core.orchestrator import *
from your_assistant.core.utils import load_env

app = Flask("Your Assistant")
cors = CORS(app)

chatgpt_orchestrator = None
rev_bard_orchestrator = None


ORCHESTRATORS = {
    "ChatGPT": ChatGPTOrchestrator,
    "RevChatGPT": RevChatGPTOrchestrator,
    "RevBard": RevBardOrchestrator,
    "QA": QAOrchestrator,
    "KnowledgeIndex": KnowledgeIndexOrchestrator,
}


# Define the function that initialize the argument parser that has the param of the prompt.
def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator")
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action="store_true",
        help="Whether to print the verbose output.",
    )

    subparsers = parser.add_subparsers(
        help="orchestrator", dest="orchestrator", required=True
    )

    for name, orchestrator in ORCHESTRATORS.items():
        subparser = subparsers.add_parser(name)
        orchestrator.add_arguments_to_parser(subparser)  # type: ignore

    return parser


def init_service():
    global chatgpt_orchestrator
    global rev_chatgpt_orchestrator
    global rev_bard_orchestrator

    load_env()

    chatgpt_orchestrator = _init_chatgpt_orchestrator()
    rev_chatgpt_orchestrator = _init_rev_chat_gpt_orchestrator()
    rev_bard_orchestrator = _init_rev_bard_orchestrator()


def _init_chatgpt_orchestrator() -> ChatGPTOrchestrator:
    parser = init_parser()
    args_to_pass = ["ChatGPT", "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return ChatGPTOrchestrator(args)


def _copy_args(source_args: argparse.Namespace, dest_args: argparse.Namespace):
    for key, value in vars(source_args).items():
        if key == "func":
            continue
        setattr(dest_args, key, value)


def _init_rev_chat_gpt_orchestrator() -> RevChatGPTOrchestrator:
    parser = init_parser()
    args_to_pass = ["RevChatGPT", "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return RevChatGPTOrchestrator(args)


def _init_rev_bard_orchestrator() -> RevBardOrchestrator:
    parser = init_parser()
    args_to_pass = ["RevBard", "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return RevBardOrchestrator(args)


@app.route("/api/v1/chatgpt", methods=["POST"])
def handle_chatgpt_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(chatgpt_orchestrator.args, runtime_args)
        response = chatgpt_orchestrator.process(args=runtime_args)
        return {"response": response}


@app.route("/api/v1/revchatgpt", methods=["POST"])
def handle_rev_chatgpt_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(rev_chatgpt_orchestrator.args, runtime_args)
        response = rev_chatgpt_orchestrator.process(args=runtime_args)
        return {"response": response}


# TODO(fuj): consider de-dup with chatgpt endpoint later.
@app.route("/api/v1/bard", methods=["POST"])
def handle_bard_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(rev_bard_orchestrator.args, runtime_args)
        response = rev_bard_orchestrator.process(args=runtime_args)
        return {"response": response}


@app.route("/api/v1/audio/transcribe", methods=["POST"])
def handle_audio_transcribe_request():
    # request.form is empty (not None) when there is no form data
    # if model not specified, default to whisper-1; otherwise, use the specified model (only whisper-1 is available as of April 2023)
    if "model" not in request.form or request.form["model"] == "whisper-1":
        model = "whisper-1"
        prompt = None
        if "prompt" in request.form:
            prompt = request.form["prompt"]
        # TODO(fuj): error handling; secure filename
        file = request.files["file"]
        file.save(file.filename)
        audio_file = open(file.filename, "rb")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # TODO(fuj): probably makes more sense to use open-source Whisper model and move this into an core audio class
        transcript = openai.Audio.transcribe(model, audio_file, prompt=prompt)

        audio_file.close()
        return {"transcript": transcript}


@app.route("/health", methods=["GET"])
def handle_health_request():
    if request.method == "GET":
        return {"response": "health success"}


if __name__ == "__main__":
    init_service()
    app.run(host="0.0.0.0", port=32167, debug=True)
