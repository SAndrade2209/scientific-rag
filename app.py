"""
Streamlit Chat Interface for Conversational RAG

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ── Make package importable ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uuid

from scientific_rag.orchestration.chat_engine import init_pipeline
from scientific_rag.utils import create_memory
from scientific_rag.config import OPENAI_API_KEY
from scientific_rag.orchestration.session_store import (
    build_session_store,
    serialize_memory,
    deserialize_memory,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Welcome to the Scientific RAG Chat",
    page_icon="🔬",
    layout="wide",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Initialization (cached — runs once)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Loading models & connecting to Qdrant...")
def _init():
    return init_pipeline(qa_model="gpt-4.1-mini", guardrail_model="gpt-4.1", expander_model="gpt-4.1-mini")

engine, POINT_COUNT = _init()

@st.cache_resource(show_spinner="Connecting to session store...")
def _init_session_store():
    return build_session_store(backend="sqlite")

session_store = _init_session_store()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Load session from persistent store (survives restarts)
session_data = session_store.load_or_create(st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = session_data.get("messages", [])

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = session_data.get("debug_logs", [])

if "memory" not in st.session_state:
    # Rebuild LangChain memory from persisted summary + buffer
    if session_data.get("buffer") or session_data.get("summary"):
        st.session_state.memory = deserialize_memory(
            data=session_data,
            api_key=OPENAI_API_KEY,
            model="gpt-4.1-mini",
            max_token_limit=1000,
        )
    else:
        st.session_state.memory = create_memory(
            api_key=OPENAI_API_KEY, model="gpt-4.1-mini", max_token_limit=1000
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar — Debug panel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.header("🔬 Scientific RAG Chat")
    st.caption(f"Collection: **{POINT_COUNT}** vectors")
    st.caption(f"Session: `{st.session_state.session_id[:12]}...`")
    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        session_store.delete(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.debug_logs = []
        st.session_state.memory = create_memory(
            api_key=OPENAI_API_KEY, model="gpt-4.1-mini", max_token_limit=1000
        )
        st.rerun()

    st.divider()
    st.subheader("🔍 Debug — Last Turn")

    if st.session_state.debug_logs:
        last = st.session_state.debug_logs[-1]

        with st.expander("🛡️ Guardrail", expanded=False):
            g = last.get("guardrail", {})
            if g:
                status = "✅ Allowed" if g["allowed"] else "🚫 Blocked"
                st.write(f"**Status:** {status}")
                st.write(f"**Reason:** {g['reason']}")
                if g.get("sanitized_query"):
                    st.write(f"**Sanitized:** {g['sanitized_query']}")

        # with st.expander("🔎 Query Expansion", expanded=False):
        #     e = last.get("expansion", {})
        #     if e:
        #         st.write(f"**Intent:** {e.get('intent', '—')}")
        #         st.write("**Sub-queries:**")
        #         for i, q in enumerate(e.get("rewritten_queries", []), 1):
        #             st.write(f"{i}. {q}")

        # with st.expander("📚 Sources", expanded=False):
        #     for s in last.get("sources", []):
        #         score = s.get("rerank_score", 0)
        #         st.write(f"• **[{score:.3f}]** {s.get('citation', '—')}")
        #
        # with st.expander("🧠 Memory Summary", expanded=False):
        #     mem_vars = st.session_state.memory.load_memory_variables({})
        #     history = mem_vars.get("history", "(empty)")
        #     st.text(history if history else "(empty)")

        # with st.expander("📋 Raw Logs", expanded=False):
        #     st.code(last.get("logs", "(no logs)"), language="text")
    else:
        st.caption("No turns yet. Ask a question to see debug info.")

    st.divider()
    st.subheader("💾 Session Persistence")

    mem_state = serialize_memory(st.session_state.memory)

    with st.expander("🔑 Session Info", expanded=False):
        st.write(f"**Session ID:** `{st.session_state.session_id}`")
        st.write(f"**Backend:** `{session_store}`")
        st.write(f"**Created:** {session_data.get('created_at', '—')}")
        st.write(f"**Updated:** {session_data.get('updated_at', '—')}")
        st.write(f"**Messages stored:** {len(session_data.get('messages', []))}")
        st.write(f"**Debug logs stored:** {len(session_data.get('debug_logs', []))}")

    with st.expander("🧠 Persisted Memory State", expanded=False):
        st.write(f"**Summary length:** {len(session_data.get('summary', ''))} chars")
        st.write(f"**Buffer messages:** {len(session_data.get('buffer', []))}")

        if session_data.get("summary"):
            st.write("**Summary (compressed older turns):**")
            st.text(session_data["summary"])
        else:
            st.caption("No summary yet — all turns still fit in the buffer.")

        # if session_data.get("buffer"):
        #     st.write("**Buffer (recent, not yet summarized):**")
        #     for m in session_data["buffer"]:
        #         prefix = "👤" if m["role"] == "user" else "🤖"
        #         st.text(f"  {prefix} {m['content'][:120]}{'...' if len(m['content']) > 120 else ''}")
        # else:
        #     st.caption("Buffer is empty — no turns yet.")

    # with st.expander("🔄 In-Memory vs Persisted", expanded=False):
    #     st.write("Compares what's in RAM right now vs what's saved in SQLite.")
    #     summary_match = mem_state["summary"] == session_data.get("summary", "")
    #     buffer_match = len(mem_state["buffer"]) == len(session_data.get("buffer", []))
    #     msgs_match = len(st.session_state.messages) == len(session_data.get("messages", []))
    #
    #     st.write(f"**Summary match:** {'✅' if summary_match else '❌'}")
    #     st.write(f"**Buffer count match:** {'✅' if buffer_match else '❌'} (RAM: {len(mem_state['buffer'])}, DB: {len(session_data.get('buffer', []))})")
    #     st.write(f"**Messages count match:** {'✅' if msgs_match else '❌'} (RAM: {len(st.session_state.messages)}, DB: {len(session_data.get('messages', []))})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main chat area
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.title("🔬 Scientific Literature — RAG Chat")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about the indexed scientific documents...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process turn
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            turn = engine.process_turn(user_input, st.session_state.memory)

        # Display answer
        display_text = f"🚫 {turn.answer}" if turn.blocked else turn.answer
        st.markdown(display_text)

    # Save to state
    st.session_state.messages.append({"role": "assistant", "content": display_text})
    st.session_state.debug_logs.append({
        "guardrail": turn.guardrail_verdict,
        "expansion": turn.expansion,
        "sources":   turn.sources,
        "logs":      turn.logs,
    })

    # ── Persist session to store ──────────────────────────────────────
    mem_state = serialize_memory(st.session_state.memory)
    session_store.save(st.session_state.session_id, {
        "messages":   st.session_state.messages,
        "summary":    mem_state["summary"],
        "buffer":     mem_state["buffer"],
        "debug_logs": st.session_state.debug_logs,
    })

    st.rerun()
