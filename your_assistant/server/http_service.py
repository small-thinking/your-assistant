"""Http service for your assistant.
"""
import os
import time
from argparse import Namespace
from typing import Type

import openai
from flask import Flask
from flask import g as app_ctx
from flask import request
from flask_cors import CORS, cross_origin

import your_assistant.core.utils as utils
from your_assistant.core.orchestrator import *
from your_assistant.core.utils import load_env

app = Flask("Your Assistant")
cors = CORS(app)

orchestrators = {}


ORCHESTRATORS = {
    "ChatGPT": ChatGPTOrchestrator,
    "Claude": AnthropicOrchestrator,
    "RevChatGPT": RevChatGPTOrchestrator,
    "RevBard": RevBardOrchestrator,
    "QA": QAOrchestrator,
}


def init_service():
    global orchestrators
    load_env()
    for name, orchestrator_type in ORCHESTRATORS.items():
        orchestrators[name] = _init_orchestrator(name, orchestrator_type)


def _copy_args(source_args: argparse.Namespace, dest_args: argparse.Namespace):
    for key, value in vars(source_args).items():
        if key == "func":
            continue
        setattr(dest_args, key, value)


def _init_orchestrator(orchestrator_name: str, orchestrator_type: Type) -> Orchestrator:
    parser = utils.init_parser(orchestrator_name, orchestrator_type)
    args_to_pass = [orchestrator_name, "--use-memory"]
    args = parser.parse_args(args_to_pass)
    return orchestrator_type(args=args)


@app.before_request
def before_request_callback():
    app.logger.debug("URL: %s, %s", request.url, request.method)
    app.logger.debug("Headers: %s", request.headers)
    app.logger.debug("Body: %s", request.get_data())
    # Store the start time for the request
    app_ctx.start_time = time.perf_counter()


@app.after_request
def after_request_callback(response):
    # Get total time in milliseconds
    total_time = time.perf_counter() - app_ctx.start_time
    time_in_ms = int(total_time * 1000)
    # Log the time taken for the endpoint
    app.logger.debug(
        "%s ms %s %s %s %s",
        time_in_ms,
        request.method,
        request.path,
        dict(request.args),
        response.get_data(),
    )
    response.headers["X-Execution-Time"] = str(time_in_ms)
    return response


@app.route("/api/v1/chatgpt", methods=["POST"])
def handle_chatgpt_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(orchestrators["ChatGPT"].args, runtime_args)
        response = orchestrators["ChatGPT"].process(args=runtime_args)
        return {"response": response}


@app.route("/api/v1/claude", methods=["POST"])
def handle_claude_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(orchestrators["Claude"].args, runtime_args)
        response = orchestrators["Claude"].process(args=runtime_args)
        return {"response": response}


@app.route("/api/v1/revchatgpt", methods=["POST"])
def handle_rev_chatgpt_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(orchestrators["RevChatGPT"].args, runtime_args)
        response = orchestrators["RevChatGPT"].process(args=runtime_args)
        return {"response": response}


# TODO(fuj): consider de-dup with chatgpt endpoint later.
@app.route("/api/v1/bard", methods=["POST"])
def handle_bard_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        runtime_args = Namespace()
        runtime_args.prompt = prompt
        _copy_args(orchestrators["RevBard"].args, runtime_args)
        response = orchestrators["RevBard"].process(args=runtime_args)
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
        _copy_args(orchestrators["QA"].args, runtime_args)
        response = orchestrators["QA"].process(args=runtime_args)
        return {"response": response}


@app.route("/health", methods=["GET"])
def handle_health_request():
    if request.method == "GET":
        return {"response": "health success"}


if __name__ == "__main__":
    init_service()
    app.run(host="0.0.0.0", port=32167, debug=True)
    app.run(host="0.0.0.0", port=32168, debug=True, ssl_context="adhoc")
