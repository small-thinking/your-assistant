# Your Assistant
Your Assistant is an **open source** AI assistant that can run on your local desktop and co-host with your private data.

Many aspect of the AI asisstant is configurable. For example, you can choose
to access it from the Discord server, the HTTP service, or locally via command line.


# Development Guide

## Project management
1. We use poetry to manage the project. Please install poetry and then use `poetry install` to install the package locally.
2. Please use `poetry add <package>` to add dependencies, or use `poetry add --group dev <package>` to add development dependencies, e.g. pytest, flake8, etc.
3. Please use black in your IDE to format your code. If you use VSCode, you can create a folder `.vscode` and create a file `settings.json`. In the file, set "format at save".
```
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
        "source.fixAll.black": true
    },
}
```

## Pre-commit
Install pre-commit:
```
poetry add --group dev pre-commit
```
and then initialize pre-commit.
```
pre-commit install
```



## Command shortcuts

### How to build the Discord service.

```
docker build -t your-assistant-discord-bot -f Dockerfile.discord . ; docker run -it your-assistant-discord-bot
```


### Clean up docker images
```
docker rm $(docker ps -a -q) ; docker images | grep '<none>' | awk '{print $3}' | xargs docker rmi
```
