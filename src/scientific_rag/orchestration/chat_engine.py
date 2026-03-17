"""
Chat Engine

Orchestrates a single conversation turn through the full pipeline:
    Guardrail → Memory → Query expansion → Multi-query retrieval → Answer → Save

This is the top-level orchestrator that keeps the Streamlit UI layer
free of business logic.
"""

from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv

from ..vectorstore.embedders import LocalEmbedder
from ..vectorstore.indexer import VectorStore
from ..retrieval.multi_query_rag import MultiQueryRAG
from .guardrail import GuardrailAgent
from .query_expander import QueryExpander
from ..utils import LogCapture
from ..config import ANSWER_PROMPT_PATH, QA_MODEL

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Turn result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Structured output of a single conversation turn."""
    answer: str = ""
    blocked: bool = False
    guardrail_verdict: dict = field(default_factory=dict)
    expansion: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    logs: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Chat engine
# ─────────────────────────────────────────────────────────────────────────────

class ChatEngine:
    """
    Stateless orchestrator for one conversation turn.

    Receives user input + memory, returns a TurnResult.
    Does not hold conversation state — that lives in the caller
    (Streamlit session_state, notebook variable, etc.).

    Components:
        - guardrail:  screens user input for safety & relevance
        - expander:   rewrites user query into multiple sub-queries
        - pipeline:   multi-query hybrid retrieval + reranking + answer generation
    """

    def __init__(
        self,
        pipeline: MultiQueryRAG,
        guardrail: GuardrailAgent,
        expander: QueryExpander,
    ):
        self.pipeline = pipeline
        self.guardrail = guardrail
        self.expander = expander

        logger.info("ChatEngine initialized")

    def process_turn(
        self,
        user_input: str,
        memory,
        top_k: int = 10,
        max_chunks: int = 50,
        candidate_multiplier: int = 3,
    ) -> TurnResult:
        """
        Execute one full conversation turn.

        Args:
            user_input:           Raw text from the user.
            memory:               LangChain ConversationSummaryBufferMemory instance.
            top_k:                Chunks to retrieve per sub-query.
            max_chunks:           Final chunks after re-reranking.
            candidate_multiplier: Candidates fetched before reranking each sub-query.

        Returns:
            TurnResult with answer, sources, debug info, and logs.
        """
        result = TurnResult()

        with LogCapture() as log_cap:
            # ── Step 0: Guardrail ─────────────────────────────────────────
            verdict = self.guardrail.check(user_input)
            result.guardrail_verdict = verdict

            if not verdict["allowed"]:
                result.blocked = True
                result.answer = verdict["response"]
                result.logs = log_cap.text
                return result

            clean_query = verdict["sanitized_query"]

            # ── Step 1: Load memory ───────────────────────────────────────
            mem_vars = memory.load_memory_variables({})
            conversation_summary = mem_vars.get("history", "")

            # ── Step 2: Expand query ──────────────────────────────────────
            expanded = self.expander.expand(clean_query, conversation_summary)
            result.expansion = expanded

            # ── Step 3: Retrieve + generate answer ────────────────────────
            rag_result = self.pipeline.ask(
                expanded,
                top_k=top_k,
                max_chunks=max_chunks,
                candidate_multiplier=candidate_multiplier,
                show_chunks=False,
                conversation_context=conversation_summary,
            )

            result.answer = rag_result["answer"]
            result.sources = rag_result.get("sources", [])

            # ── Step 4: Save to memory ────────────────────────────────────
            memory.save_context(
                {"input": clean_query},
                {"output": result.answer},
            )

        result.logs = log_cap.text
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Factory — creates all heavy components once
# ─────────────────────────────────────────────────────────────────────────────

def init_pipeline(
    qa_model: str = QA_MODEL,
    guardrail_model: str = "gpt-4.1",
    expander_model: str = "gpt-4.1-mini",
) -> tuple:
    """
    Initialise all heavy components (embedder, store, reranker, etc.).

    Returns:
        (engine: ChatEngine, point_count: int)
    """
    answer_prompt = ANSWER_PROMPT_PATH.read_text(encoding="utf-8")

    embedder  = LocalEmbedder(model_name="BAAI/bge-m3")
    store     = VectorStore(embedder=embedder, use_hybrid=True)
    pipeline  = MultiQueryRAG(store=store, answer_prompt=answer_prompt, model=qa_model, reranker_model= "BAAI/bge-reranker-base")
    guardrail = GuardrailAgent(model=guardrail_model)
    expander  = QueryExpander(model=expander_model, temperature=0.3)
    engine    = ChatEngine(pipeline=pipeline, guardrail=guardrail, expander=expander)

    point_count = store.collection_info().points_count

    return engine, point_count

