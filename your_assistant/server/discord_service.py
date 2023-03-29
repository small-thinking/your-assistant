"""Create the discord service.
"""
import asyncio
import os

import discord
from discord.ext import commands

from your_assistant.core.orchestrator import RevChatGPTOrchestrator
from your_assistant.core.utils import Logger, load_env


class DiscordBot(commands.Bot):
    def __init__(self) -> None:
        """Initialize the bot."""
        load_env()
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="/", intents=intents)
        # Init the orchestrator.
        self.logger = Logger("DiscordBot")
        self.activity = discord.Activity(
            type=discord.ActivityType.listening, name="/speak"
        )
        self.orchestrator = RevChatGPTOrchestrator(verbose=True)

    async def on_ready(self):
        """When the bot is ready."""
        self.logger.info(f"Logged in as {self.user}")
        await self.change_presence(activity=self.activity)

    async def on_message(self, message):
        """When a message is received."""
        if message.author == self.user:
            return
        await message.channel.typing()
        author, channel, content = message.author, message.channel, message.content
        self.logger.info(
            f"Received message from {author} in {channel}: {message.content}"
        )
        response = self.orchestrator.process(prompt=content)
        await message.channel.send(response)


def main():
    # Create the bot.
    bot = DiscordBot()

    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    if DISCORD_TOKEN is None:
        raise ValueError("DISCORD_TOKEN is not set.")

    # Run the bot.
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
