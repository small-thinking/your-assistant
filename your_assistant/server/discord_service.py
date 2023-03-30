"""Create the discord service.
"""
import asyncio
import os

import discord
from discord import app_commands
from discord.ext import commands

from your_assistant.core.orchestrator import *
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
            type=discord.ActivityType.listening, name="/bard or /chat"
        )
        self.chat_orchestrator = RevChatGPTOrchestrator(verbose=True)
        self.bard_orchestrator = RevBardOrchestrator(verbose=True)

    async def on_ready(self):
        """When the bot is ready."""
        try:
            self.logger.info(f"Ready! Logged in as {self.user}")
            synced = await self.tree.sync()
            self.logger.info(f"Synced {len(synced)} commands.")
            await self.change_presence(activity=self.activity)
        except Exception as e:
            self.logger.error(f"Failed to sync application commands: {e}")


bot = DiscordBot()


@bot.tree.command(name="chat")
@app_commands.describe(prompt="prompt")
async def chat(interaction: discord.Interaction, prompt: str):
    """Speak to the ChatGPT bot."""
    await speak_to_bot(interaction, prompt, bot.chat_orchestrator)


@bot.tree.command(name="bard")
@app_commands.describe(prompt="prompt")
async def bard(interaction: discord.Interaction, prompt: str):
    """Speak to the Bard bot."""
    await speak_to_bot(interaction, prompt, bot.bard_orchestrator)


async def speak_to_bot(
    interaction: discord.Interaction, prompt: str, orchestrator: Orchestrator
):
    """Speak to the bot."""
    try:
        await interaction.channel.typing()
        await interaction.response.defer()
        user, channel = (
            interaction.user.name,
            interaction.channel,
        )
        bot.logger.info(
            f"Received message from {user} in channel [{channel}]: {prompt}"
        )
        response = orchestrator.process(prompt=prompt)
        user_mention = interaction.user.mention
        response = f"{user_mention} {response}"
        await interaction.followup.send(response)
    except Exception as e:
        bot.logger.error(f"Failed to send message: {e}")


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if DISCORD_TOKEN is None:
    raise ValueError("DISCORD_TOKEN is not set.")

# Run the bot.
bot.run(DISCORD_TOKEN)
