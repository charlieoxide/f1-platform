"""Test isolation: the suite must give the same result on every machine.

Two tests failed on one developer's laptop only, because his local .env set
F1_API_KEY and the API then (correctly) demanded a key the tests never sent.
python-dotenv never overrides variables that already exist, so setting them
here, before src.config is imported anywhere, beats whatever a local .env says.
Tests that need a key set one explicitly.
"""

import os

os.environ["F1_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_MODEL"] = "openrouter/free"
os.environ["TELEGRAM_BOT_TOKEN"] = ""
