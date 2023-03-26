"""Utilities.
"""
from dotenv import load_dotenv

class Config:
    def __init__(self, env_file_path: str = None):
        load_dotenv(env_file_path)
