"""
Simple test script to verify retriever_rerank.py works correctly
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Allow running from the tests/ directory or from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag import LocalEmbedder, VectorStore, RAGPipelineWithReranking
from scientific_rag.config import ANSWER_PROMPT_PATH

load_dotenv()

print("="*80)
print("Testing Reranking Enhancement")
print("="*80)

ANSWER_PROMPT = ANSWER_PROMPT_PATH.read_text(encoding="utf-8")

# Initialize components
print("\n1. Initializing embedder...")
embedder = LocalEmbedder()

print("2. Connecting to vector store...")
store = VectorStore(embedder=embedder, use_hybrid=True)

print("3. Initializing RAG pipeline with reranking...")
rag_rerank = RAGPipelineWithReranking(
    store=store,
    answer_prompt=ANSWER_PROMPT,
    model="gpt-4.1-mini"
)

print("\n✓ All components initialized successfully!\n")

# Test question
test_question = "What methods are used to remove rubber deposits from runway surfaces?"

print(f"Test Question: {test_question}\n")
print("="*80)
print("Running retrieval with reranking...")
print("="*80)

result = rag_rerank.ask(
    question=test_question,
    top_k=5,
    show_chunks=True,
    candidate_multiplier=2
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nAnswer:\n{result['answer']}\n")

print("Sources:")
for i, s in enumerate(result['sources'], 1):
    print(f"  {i}. {s['title']} ({s['year']})")
    print(f"     Rerank score: {s['rerank_score']:.3f}, Original score: {s['original_score']:.3f}")

print(f"\nTotal chunks used: {len(result['chunks'])}")
print("\nTop 3 chunks:")
for i, c in enumerate(result['chunks'][:3], 1):
    title = c['metadata'].get('title', 'Unknown')[:50]
    section = c['metadata'].get('section_h2', 'N/A')[:30]
    print(f"  {i}. [Rerank: {c['rerank_score']:.3f}, Original: {c['original_score']:.3f}]")
    print(f"     {title}")
    print(f"     Section: {section}")

print("\n" + "="*80)
print("✓ Test completed successfully!")
print("="*80)
