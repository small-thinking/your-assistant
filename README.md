# Your Assistant
Your Assistant is an **open source** AI assistant that can run on your local desktop and co-host with your private data.

Many aspect of the AI asisstant is configurable. For example, you can choose
to access it from the Discord server, the HTTP service, or locally via command line.


# Development Guide

## Project management
We use the following tools to east the project development:
1. poetry: package management.
2. mypy: static type check.
3. flake8 and black: coding style unification and formatting.
4. isort: import order management.

Please run `dev-setup.sh` to setup the environment.

When you need to add new package, please use `poetry add <package>` to add dependencies, or use `poetry add --group dev <package>` to add development dependencies, e.g. pytest, flake8, etc.
For auto-formatting, please use black in your IDE to format your code. If you use VSCode, you can create a folder `.vscode` and create a file `settings.json`. In the file, set "format at save".
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
We have a number of pre-commit checks, including coding style, type check, and import order,

Initialize pre-commit.
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
