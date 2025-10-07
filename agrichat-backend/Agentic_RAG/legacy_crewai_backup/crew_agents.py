"""
Archived CrewAI agents (backup). Not imported by default.
"""

from tools import RAGTool, FallbackAgriTool
import os
from dotenv import load_dotenv
try:
    from crewai import LLM, Agent
except Exception:
    # Placeholder: crewai not required for archived copy
    class Agent:
        def __init__(self, *args, **kwargs):
            pass

    class LLM:
        def __init__(self, *args, **kwargs):
            pass

from typing import List, Dict, Optional

load_dotenv()

# (Archived content omitted)
