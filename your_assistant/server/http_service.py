"""Http service for your assistant.
"""
from flask import Flask, request

from your_assistant.core.orchestrator import *
from your_assistant.core.utils import load_env

app = Flask("Your Assistant")
chatgpt_orchestrator = None
bard_orchestrator = None

def init_service():
    global chatgpt_orchestrator
    global bard_orchestrator

    load_env()

    chatgpt_orchestrator = RevChatGPTOrchestrator()
    bard_orchestrator = RevBardOrchestrator()


@app.route("/api/v1/chat", methods=["POST"])
def handle_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        response = chatgpt_orchestrator.process(prompt=prompt)
        return {"response": response}

# TODO(fuj): consider de-dup with chatgpt endpoint later.
@app.route("/api/v1/bard", methods=["POST"])
def handle_bard_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        response = bard_orchestrator.process(prompt=prompt)
        return {"response": response}

@app.route("/health", methods=["GET"])
def handle_health_request():
    if request.method == "GET":
        return {"response": "health success"}

if __name__ == "__main__":
    init_service()
    app.run(host="0.0.0.0", port=32167, debug=True)
