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
bard_orchestrator = None


def init_service():
    global chatgpt_orchestrator
    global bard_orchestrator

    load_env()

    chatgpt_orchestrator = RevChatGPTOrchestrator()
    bard_orchestrator = RevBardOrchestrator()


@app.route("/api/v1/chat", methods=["POST"])
def handle_chatgpt_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        args = Namespace()
        args.prompt = prompt
        response = chatgpt_orchestrator.process(args=args)
        return {"response": response}


# TODO(fuj): consider de-dup with chatgpt endpoint later.
@app.route("/api/v1/bard", methods=["POST"])
def handle_bard_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        args = Namespace()
        args.prompt = prompt
        response = bard_orchestrator.process(args=args)
        return {"response": response}


@app.route("/api/v1/audio/transcribe", methods=["POST"])
def handle_audio_transcribe_request():
    # request.form is empty (not None) when there is no form data
    # if model not specified, default to whisper-1; otherwise, use the specified model (only whisper-1 is available as of April 2023)
    if "model" not in request.form or request.form["model"] == "whisper-1":
        model = "whisper-1"
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
