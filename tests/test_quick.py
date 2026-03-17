"""
Script de prueba rápida del sistema RAG
Ejecuta un subconjunto representativo de preguntas
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Allow running from the tests/ directory or from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scientific_rag import LocalEmbedder, VectorStore, RAGPipeline, RAGPipelineWithReranking
from scientific_rag.config import ANSWER_PROMPT_PATH, QA_MODEL
from test_questions_comprehensive import test_questions

load_dotenv()

ANSWER_PROMPT = ANSWER_PROMPT_PATH.read_text(encoding="utf-8")

print("="*80)
print("TEST RÁPIDO DEL SISTEMA RAG")
print("="*80)

# Inicializar
print("\n1. Inicializando componentes...")
embedder = LocalEmbedder()
store = VectorStore(embedder=embedder, use_hybrid=True)

rag_original = RAGPipeline(store=store, answer_prompt=ANSWER_PROMPT, model=QA_MODEL)
rag_rerank = RAGPipelineWithReranking(store=store, answer_prompt=ANSWER_PROMPT, model=QA_MODEL)

print("✓ Retrievers listos")

# Seleccionar preguntas representativas
print("\n2. Seleccionando preguntas de prueba...")

test_set = [
    # Pregunta específica de 1 documento
    next(q for q in test_questions if q['id'] == 'Q2_SPECIFIC'),
    # Pregunta multi-documento
    next(q for q in test_questions if q['id'] == 'Q6_MULTI'),
    # Pregunta técnica
    next(q for q in test_questions if q['id'] == 'Q15_TECHNICAL'),
    # Pregunta fuera de alcance
    next(q for q in test_questions if q['id'] == 'Q25_OUT_OF_SCOPE'),
]

print(f"✓ {len(test_set)} preguntas seleccionadas")

# Ejecutar pruebas
results = []

for i, q_data in enumerate(test_set, 1):
    print(f"\n{'='*80}")
    print(f"PREGUNTA {i}/{len(test_set)} - {q_data['id']}")
    print(f"Categoría: {q_data['category']}")
    print(f"{'='*80}")
    print(f"\nQ: {q_data['question']}")
    print(f"\n{'-'*80}")

    # Original
    print("\n🔵 ORIGINAL:")
    start = time.time()
    result_orig = rag_original.ask(q_data['question'], top_k=5, show_chunks=True)
    time_orig = time.time() - start

    docs_orig = set(c['metadata'].get('stem', '') for c in result_orig['chunks'])

    print(f"  Tiempo: {time_orig:.2f}s")
    print(f"  Documentos: {len(docs_orig)}")
    print(f"  Respuesta: {result_orig['answer'][:200]}...")

    # Reranking
    print("\n🟢 RERANKING:")
    start = time.time()
    result_rerank = rag_rerank.ask(q_data['question'], top_k=5, show_chunks=True, candidate_multiplier=2)
    time_rerank = time.time() - start

    docs_rerank = set(c['metadata'].get('stem', '') for c in result_rerank['chunks'])

    print(f"  Tiempo: {time_rerank:.2f}s (+{time_rerank - time_orig:.2f}s)")
    print(f"  Documentos: {len(docs_rerank)}")
    print(f"  Respuesta: {result_rerank['answer'][:200]}...")

    # Comparación
    shared_docs = docs_orig & docs_rerank
    print(f"\n📊 COMPARACIÓN:")
    print(f"  Documentos compartidos: {len(shared_docs)}")
    print(f"  Solo en original: {len(docs_orig - docs_rerank)}")
    print(f"  Solo en reranking: {len(docs_rerank - docs_orig)}")

    # Verificar términos esperados
    expected_terms = q_data.get('answer_should_include', [])
    if expected_terms:
        found_orig = sum(1 for term in expected_terms if term.lower() in result_orig['answer'].lower())
        found_rerank = sum(1 for term in expected_terms if term.lower() in result_rerank['answer'].lower())
        print(f"\n✓ Términos clave encontrados:")
        print(f"  Original: {found_orig}/{len(expected_terms)}")
        print(f"  Reranking: {found_rerank}/{len(expected_terms)}")

    results.append({
        'q_id': q_data['id'],
        'category': q_data['category'],
        'time_orig': time_orig,
        'time_rerank': time_rerank,
        'docs_orig': len(docs_orig),
        'docs_rerank': len(docs_rerank),
    })

# Resumen final
print(f"\n{'='*80}")
print("RESUMEN FINAL")
print(f"{'='*80}")

avg_time_orig = sum(r['time_orig'] for r in results) / len(results)
avg_time_rerank = sum(r['time_rerank'] for r in results) / len(results)

print(f"\nTiempo promedio:")
print(f"  Original:  {avg_time_orig:.2f}s")
print(f"  Reranking: {avg_time_rerank:.2f}s")
print(f"  Overhead:  +{avg_time_rerank - avg_time_orig:.2f}s ({((avg_time_rerank/avg_time_orig - 1)*100):.1f}%)")

print(f"\nDocumentos promedio:")
avg_docs_orig = sum(r['docs_orig'] for r in results) / len(results)
avg_docs_rerank = sum(r['docs_rerank'] for r in results) / len(results)
print(f"  Original:  {avg_docs_orig:.1f}")
print(f"  Reranking: {avg_docs_rerank:.1f}")

print(f"\n{'='*80}")
print("✓ TEST COMPLETADO")
print(f"{'='*80}")
print(f"\nPara test completo, ejecuta: jupyter notebook test_comprehensive.ipynb")
