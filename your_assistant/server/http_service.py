"""Http service for your assistant.
"""
import os
from argparse import Namespace

import openai
from flask import Flask, request
from flask_cors import CORS, cross_origin

import your_assistant.core.utils as utils
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


def init_service():
    global chatgpt_orchestrator
    global rev_chatgpt_orchestrator
    global rev_bard_orchestrator
    global qa_orchestrator

    load_env()

    chatgpt_orchestrator = _init_chatgpt_orchestrator()
    rev_chatgpt_orchestrator = _init_rev_chat_gpt_orchestrator()
    rev_bard_orchestrator = _init_rev_bard_orchestrator()
    qa_orchestrator = _init_qa_orchestrator()


def _copy_args(source_args: argparse.Namespace, dest_args: argparse.Namespace):
    for key, value in vars(source_args).items():
        if key == "func":
            continue
        setattr(dest_args, key, value)


def _init_chatgpt_orchestrator() -> ChatGPTOrchestrator:
    parser = utils.init_parser(ORCHESTRATORS)
    args_to_pass = ["ChatGPT", "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return ChatGPTOrchestrator(args)


def _init_rev_chat_gpt_orchestrator() -> RevChatGPTOrchestrator:
    parser = utils.init_parser(ORCHESTRATORS)
    args_to_pass = ["RevChatGPT", "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return RevChatGPTOrchestrator(args)


def _init_rev_bard_orchestrator() -> RevBardOrchestrator:
    parser = utils.init_parser(ORCHESTRATORS)
    args_to_pass = ["RevBard", "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return RevBardOrchestrator(args)


def _init_qa_orchestrator() -> QAOrchestrator:
    parser = utils.init_parser(ORCHESTRATORS)
    args_to_pass = [
        "QA",
        "--use-memory",
        "--max-token-size",
        "800",
        "--memory-token-size",
        "300",
    ]
    args = parser.parse_args(args_to_pass)
    return QAOrchestrator(args)


@app.before_request
def before_request_callback():
    app.logger.debug("URL: %s, %s", request.url, request.method)
    app.logger.debug("Headers: %s", request.headers)
    app.logger.debug("Body: %s", request.get_data())


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


@app.route("/api/v1/qa", methods=["POST"])
def handle_qa_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(qa_orchestrator.args, runtime_args)
        response = qa_orchestrator.process(args=runtime_args)
        return {"response": response}


@app.route("/health", methods=["GET"])
def handle_health_request():
    if request.method == "GET":
        return {"response": "health success"}


if __name__ == "__main__":
    init_service()
    app.run(host="0.0.0.0", port=32167, debug=True)
