# Your Assistant 
Your Assistant is an **open source** AI assistant that can run on your local desktop and co-host with your private data.

Many aspect of the AI asisstant is configurable. For example, you can choose
to access it from the Discord server, the HTTP service, or locally via command line.

## How to build the Discord service.

`docker build -t your-assistant-discord-bot -f Dockerfile.discord .`

`docker run -it your-assistant-discord-bot`

docker rm $(docker ps -a -q) ; docker images | grep '<none>' | awk '{print $3}' | xargs docker rmi