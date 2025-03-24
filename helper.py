import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path("credentials/.env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

print(api_key)