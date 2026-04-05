"""Те же переменные окружения и дефолты, что в academiy_test_llm_safety/backend/app/config.py."""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_MODEL = "grok-4.20-non-reasoning"

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
XAI_MODEL = os.getenv("XAI_MODEL", DEFAULT_XAI_MODEL)
