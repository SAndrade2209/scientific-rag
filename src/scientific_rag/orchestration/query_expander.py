"""
Query Expander

Uses an LLM to rewrite a user query into multiple specific,
semantically rich sub-queries for vector search. Supports conversation
context so follow-up questions can be resolved.
"""

import json
from loguru import logger
from openai import OpenAI
from os import environ
from dotenv import load_dotenv

from ..config import QUERY_EXPANSION_CONV_PROMPT_PATH

load_dotenv()


class QueryExpander:
    """
    Rewrites a user query into multiple search-optimised sub-queries.

    When conversation_summary is provided, the LLM uses it to resolve
    pronouns and follow-up references (e.g. "tell me more about that").

    Returns a dict:
        {
            "original_query": "...",
            "intent": "...",
            "rewritten_queries": ["q1", "q2", "q3"]
        }
    """

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.3):
        self.client = OpenAI(api_key=environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.system_prompt = QUERY_EXPANSION_CONV_PROMPT_PATH.read_text(encoding="utf-8")
        logger.info(f"QueryExpander initialized with model: {model}")

    def expand(self, user_query: str, conversation_summary: str = "") -> dict:
        """
        Expand a user query into multiple sub-queries.

        Args:
            user_query:            The raw question from the user.
            conversation_summary:  Summary of previous turns (optional).

        Returns:
            dict with keys: original_query, intent, rewritten_queries
        """
        logger.info(f"Expanding query: {user_query[:80]}...")

        if conversation_summary:
            user_message = (
                f"CONVERSATION CONTEXT:\n{conversation_summary}\n\n"
                f"USER QUESTION:\n{user_query}"
            )
        else:
            user_message = user_query

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )

        result = json.loads(response.choices[0].message.content)

        logger.info(f"Intent: {result.get('intent', '—')}")
        logger.info(f"Generated {len(result.get('rewritten_queries', []))} sub-queries")

        return result

