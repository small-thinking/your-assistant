"""Utilities.
"""
from dotenv import load_dotenv


def load_env(env_file_path: str = None):
    load_dotenv(env_file_path)
