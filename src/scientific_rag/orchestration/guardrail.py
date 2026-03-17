"""
Guardrail Agent

Evaluates user input for safety, policy compliance, and domain relevance
before it reaches the retrieval pipeline.

Returns a structured verdict that the chat loop uses to either proceed
or show a polite refusal message.
"""

import json
from loguru import logger
from openai import OpenAI
from os import environ
from dotenv import load_dotenv

from ..config import GUARDRAIL_PROMPT_PATH

load_dotenv()


class GuardrailAgent:
    """
    Screens user input before it enters the RAG pipeline.

    Checks for:
      - Harmful / abusive / illegal content
      - Jailbreak and prompt injection attempts
      - Off-topic queries unrelated to the domain
      - Manipulative preambles that should be stripped

    Returns a dict with:
      - allowed:         bool — whether the query can proceed
      - reason:          str  — internal reason for the decision
      - response:        str  — polite refusal message (empty if allowed)
      - sanitized_query: str  — cleaned query to pass forward (empty if blocked)
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=environ["OPENAI_API_KEY"])
        self.model = model
        self.system_prompt = GUARDRAIL_PROMPT_PATH.read_text(encoding="utf-8")
        logger.info(f"GuardrailAgent initialized with model: {model}")

    def check(self, user_input: str) -> dict:
        """
        Evaluate a user message for safety and relevance.

        Args:
            user_input: The raw message from the user.

        Returns:
            dict with keys: allowed, reason, response, sanitized_query
        """
        logger.info(f"Guardrail checking: {user_input[:80]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_input},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result = json.loads(response.choices[0].message.content)

            # Ensure all expected keys exist
            verdict = {
                "allowed":         result.get("allowed", True),
                "reason":          result.get("reason", ""),
                "response":        result.get("response", ""),
                "sanitized_query": result.get("sanitized_query", user_input),
            }

        except Exception as e:
            # Fail-open: if the guardrail itself fails, let the query through
            logger.warning(f"Guardrail error (fail-open): {e}")
            verdict = {
                "allowed": True,
                "reason": f"Guardrail error: {e}",
                "response": "",
                "sanitized_query": user_input,
            }

        status = "✅ ALLOWED" if verdict["allowed"] else "🚫 BLOCKED"
        logger.info(f"Guardrail verdict: {status} — {verdict['reason']}")

        return verdict

