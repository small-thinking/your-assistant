"""Create the discord service.
"""
import os

import discord
from discord.ext import commands

from your_assistant.core.orchestrator import RevChatGPTOrchestrator
from your_assistant.core.utils import Logger, load_env


# Implement a discord bot that accept the user input with /speaks command.
# The bot should be able to send the user input to the orchestrator and send the response back to the user.
class DiscordBot:
    def __init__(self):
        self.logger = Logger("DiscordBot")
        load_env()
        intents = discord.Intents.default()
        intents.message_content = True
        # Set up the bot client.
        self.client = discord.Client(intents=intents)
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

        self.orchestrator = RevChatGPTOrchestrator(verbose=True)

    async def on_ready(self):
        """Log when the bot is ready."""
        self.logger.info("Bot is ready.")

    async def on_message(self, message):
        """Reply the message"""
        if message.author == self.client.user:
            return

        if message.content.startswith("/speak"):
            await message.channel.send("Hello!")

    @commands.command(name="speak", help="Speak to the bot.")
    async def speak(self, ctx, *args):
        """Speak to the agent.

        Args:
            ctx (commands.Context): The context of the command.
            *args (str): The arguments of the command.
        """
        await ctx.trigger_typing()

        message = ctx.message
        author, channel = ctx.author, ctx.channel
        self.logger.info(f"Received message from {author} in {channel}: {message}")
        response = self.orchestrator.process(prompt=message)
        await ctx.send(response)


if __name__ == "__main__":
    # Create the bot.
    bot = DiscordBot()

    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    if DISCORD_TOKEN is None:
        raise ValueError("DISCORD_TOKEN is not set.")

    # Run the bot.
    bot.client.run(DISCORD_TOKEN)
