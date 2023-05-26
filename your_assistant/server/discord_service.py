# type: ignore
"""Create the discord service.
"""
import argparse
import os
import traceback
from typing import Type

import discord
from discord import app_commands
from discord.ext import commands

from your_assistant.core.orchestrator import *
from your_assistant.core.utils import Logger, init_parser, load_env

AVAILABLE_ORCHESTRATORS = {
    "ChatGPT": ChatGPTOrchestrator,
    "Claude": AnthropicOrchestrator,
    "RevBard": RevBardOrchestrator,
    "PaLM": PaLMOrchestrator,
    "QA": QAOrchestrator,
}

orchestrators = {}


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

        for name, orchestrator_type in AVAILABLE_ORCHESTRATORS.items():
            self.logger.info(f"Initializing {name}...")
            orchestrators[name] = self._init_orchestrator(name, orchestrator_type)

    def _init_orchestrator(
        self,
        orchestrator_name: str,
        orchestrator_type: Type,
    ) -> Orchestrator:
        parser = init_parser(orchestrator_name, orchestrator_type)
        args_to_pass = [orchestrator_name, "--use-memory"]
        args = parser.parse_args(args_to_pass)
        return orchestrator_type(args=args)

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
async def chat(interaction: discord.Interaction, prompt: str) -> None:
    """Speak to the ChatGPT bot."""
    args = argparse.Namespace()
    args.prompt = prompt
    args.use_memory = True
    await speak_to_bot(interaction, args, "ChatGPT", orchestrators["ChatGPT"])


@bot.tree.command(name="claude")
@app_commands.describe(prompt="prompt")
async def claude(interaction: discord.Interaction, prompt: str) -> None:
    """Speak to the Claude bot."""
    args = argparse.Namespace()
    args.prompt = prompt
    args.use_memory = True
    await speak_to_bot(interaction, args, "Claude", orchestrators["Claude"])


@bot.tree.command(name="bard")
@app_commands.describe(prompt="prompt")
async def bard(interaction: discord.Interaction, prompt: str) -> None:
    """Speak to the Bard bot."""
    args = argparse.Namespace()
    args.prompt = prompt
    args.use_memory = True
    await speak_to_bot(interaction, args, "Bard", orchestrators["RevBard"])


@bot.tree.command(name="palm")
@app_commands.describe(prompt="prompt")
async def chat(interaction: discord.Interaction, prompt: str) -> None:
    """Speak to the PaLM bot."""
    args = argparse.Namespace()
    args.prompt = prompt
    args.use_memory = True
    await speak_to_bot(interaction, args, "PaLM", orchestrators["PaLM"])


@bot.tree.command(name="qa")
@app_commands.describe(prompt="prompt")
async def qa(interaction: discord.Interaction, prompt: str) -> None:
    """Speak to the QA bot."""
    args = argparse.Namespace()
    args.prompt = prompt
    args.use_memory = True
    await speak_to_bot(interaction, args, "QA", orchestrators["QA"])


async def speak_to_bot(
    interaction: discord.Interaction,
    args: argparse.Namespace,
    orchestrator_name: str,
    orchestrator: Orchestrator,
) -> None:
    """Speak to the bot."""
    try:
        await interaction.channel.typing()
        await interaction.response.defer()
        user, channel = (
            interaction.user.name,
            interaction.channel,
        )
        bot.logger.info(
            f"Received message from {user} in channel [{channel}]: {args.prompt}"
        )
        await interaction.followup.send(f"To {orchestrator_name}: {args.prompt}...")
        response = orchestrator.process(args=args)
        user_mention = interaction.user.mention
        response = f"{user_mention} {response}"
        bot.logger.info(f"Sending response: {response}")
        await interaction.followup.send(response)
    except Exception as e:
        error_message = f"Failed to send message: {e}.\n{traceback.format_exc()}"
        bot.logger.error(error_message)
        await interaction.followup.send(error_message)


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if DISCORD_TOKEN is None:
    raise ValueError("DISCORD_TOKEN is not set.")

# Run the bot.
bot.run(DISCORD_TOKEN)
