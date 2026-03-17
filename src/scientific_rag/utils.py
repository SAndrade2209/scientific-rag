"""
Shared utilities for the scientific_rag package.

Contains:
    - LogCapture:     Context manager to capture loguru output into a buffer
    - create_memory:  Factory to initialise LangChain conversation memory
"""

from io import StringIO
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Log capture
# ─────────────────────────────────────────────────────────────────────────────

class LogCapture:
    """
    Context manager that temporarily adds a loguru sink to capture
    all log output into a string buffer.

    Usage:
        with LogCapture() as cap:
            ...  # any code that calls logger.info / logger.debug
        print(cap.text)   # captured log output
    """

    def __enter__(self):
        self.buffer = StringIO()
        self.sink_id = logger.add(
            self.buffer,
            format="{time:HH:mm:ss} | {level:<7} | {message}",
            level="DEBUG",
        )
        return self

    def __exit__(self, *args):
        logger.remove(self.sink_id)

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation memory factory
# ─────────────────────────────────────────────────────────────────────────────

def create_memory(
    api_key: str,
    model: str = "gpt-4.1-mini",
    max_token_limit: int = 1200,
):
    """
    Create a LangChain ConversationSummaryBufferMemory instance.

    Keeps recent messages verbatim and summarises older ones when the
    buffer exceeds ``max_token_limit`` tokens.

    Args:
        api_key:          OpenAI API key for the summarisation LLM.
        model:            Model used for summarisation.
        max_token_limit:  Token budget before older messages are summarised.

    Returns:
        ConversationSummaryBufferMemory
    """
    from langchain_classic.memory import ConversationSummaryBufferMemory
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        human_prefix="User",
        ai_prefix="Assistant",
    )

