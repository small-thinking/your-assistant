"""Http service for your assistant.
"""
from flask import Flask, request

from your_assistant.core.orchestrator import RevChatGPTOrchestrator
from your_assistant.core.utils import load_env

app = Flask("Your Assistant")
orchestrator = None


def init_service():
    global orchestrator
    load_env()

    orchestrator = RevChatGPTOrchestrator()


@app.route("/api/v1/assistant", methods=["POST"])
def handle_request():
    if request.method == "POST":
        prompt = request.json["prompt"]
        response = orchestrator.process(prompt=prompt)
        return {"response": response}

@app.route("/health", methods=["GET"])
def handle_health_request():
    if request.method == "GET":
        return {"response": "health success"}

if __name__ == "__main__":
    init_service()
    app.run(host="0.0.0.0", port=32167, debug=True)
