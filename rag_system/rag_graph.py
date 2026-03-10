"""
LangGraph 기반 고급 RAG 워크플로우
- 상태 기반 그래프로 RAG 파이프라인 구성
- 노드: 쿼리 분석 → 검색 → 문서 그레이딩 → 생성 → 환각 검증 → 답변 검증
- 조건부 엣지로 쿼리 재작성, 재생성 등 자동 제어
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from rag_chain import (
    ChromaRetriever,
    format_docs_as_context,
    get_llm,
    rewrite_query,
    grade_document_relevance,
    check_hallucination,
    RAG_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser
from config import (
    OPENAI_API_KEY,
    TOP_K,
    MAX_QUERY_REWRITES,
    ENABLE_HALLUCINATION_CHECK,
    MAX_GENERATION_RETRIES,
)


# ============================================================
# 상태 정의 (LangGraph State)
# ============================================================
class RAGState(TypedDict):
    """RAG 그래프의 전역 상태"""
    question: str                        # 원본 질문
    rewritten_query: str                 # 재작성된 쿼리
    retrieved_docs: list[dict]           # 검색된 문서 리스트
    relevant_docs: list[dict]            # 관련성 필터링된 문서 리스트
    context: str                         # LLM에 전달할 컨텍스트
    answer: str                          # 생성된 답변
    sources: list[dict]                  # 출처 정보
    rewrite_count: int                   # 쿼리 재작성 횟수
    generation_count: int                # 답변 생성 횟수
    is_grounded: bool                    # 환각 검증 결과
    steps: list[str]                     # 실행된 노드 추적
    researcher: str | None               # 연구원 필터
    top_k: int                           # 검색 결과 수


# ============================================================
# 그래프 노드 (각 단계)
# ============================================================

def analyze_query(state: RAGState) -> dict:
    """Step 1: 쿼리 분석 및 재작성"""
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count == 0:
        rewritten = rewrite_query(question)
    else:
        rewritten = rewrite_query(state.get("rewritten_query", question))

    return {
        "rewritten_query": rewritten,
        "rewrite_count": rewrite_count + 1,
        "steps": state.get("steps", []) + [f"쿼리 분석 (재작성 #{rewrite_count + 1}: '{rewritten}')"],
    }


def retrieve_documents(state: RAGState) -> dict:
    """Step 2: 벡터 검색"""
    query = state.get("rewritten_query", state["question"])
    top_k = state.get("top_k", TOP_K)
    researcher = state.get("researcher")

    retriever = ChromaRetriever(top_k=top_k, researcher=researcher)
    docs = retriever.invoke(query)

    return {
        "retrieved_docs": docs,
        "steps": state.get("steps", []) + [f"벡터 검색 완료 ({len(docs)}건 검색됨)"],
    }


def grade_documents(state: RAGState) -> dict:
    """Step 3: 검색된 문서의 관련성 평가 (LLM 기반)"""
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if not OPENAI_API_KEY:
        return {
            "relevant_docs": docs,
            "steps": state.get("steps", []) + ["문서 그레이딩 스킵 (API 키 없음)"],
        }

    relevant = []
    for doc in docs:
        if grade_document_relevance(question, doc["text"]):
            relevant.append(doc)

    return {
        "relevant_docs": relevant,
        "steps": state.get("steps", []) + [
            f"문서 관련성 평가 ({len(relevant)}/{len(docs)}건 관련)"
        ],
    }


def generate_answer(state: RAGState) -> dict:
    """Step 4: 관련 문서 기반 답변 생성"""
    question = state["question"]
    relevant_docs = state.get("relevant_docs", [])
    generation_count = state.get("generation_count", 0)

    context = format_docs_as_context(relevant_docs)

    sources = [
        {
            "file_name": r["metadata"]["file_name"],
            "researcher": r["metadata"]["researcher"],
            "page_num": r["metadata"]["page_num"],
            "score": r["score"],
            "snippet": r["text"][:150],
        }
        for r in relevant_docs
    ]

    if not OPENAI_API_KEY:
        answer = f"[LLM 미사용 - API 키 없음]\n\n{context}"
    else:
        try:
            llm = get_llm()
            chain = RAG_PROMPT | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
        except Exception as e:
            answer = f"[LLM 오류] {e}\n\n{context}"

    return {
        "context": context,
        "answer": answer,
        "sources": sources,
        "generation_count": generation_count + 1,
        "steps": state.get("steps", []) + [f"답변 생성 (#{generation_count + 1})"],
    }


def check_hallucination_node(state: RAGState) -> dict:
    """Step 5: 환각 검증"""
    context = state.get("context", "")
    answer = state.get("answer", "")

    if not ENABLE_HALLUCINATION_CHECK or not OPENAI_API_KEY:
        return {
            "is_grounded": True,
            "steps": state.get("steps", []) + ["환각 검증 스킵"],
        }

    is_grounded = check_hallucination(context, answer)

    return {
        "is_grounded": is_grounded,
        "steps": state.get("steps", []) + [
            f"환각 검증: {'통과 (grounded)' if is_grounded else '실패 (hallucinated)'}"
        ],
    }


def build_final_response(state: RAGState) -> dict:
    """Step 6: 최종 응답 구성"""
    return {
        "steps": state.get("steps", []) + ["최종 응답 구성 완료"],
    }


# ============================================================
# 조건부 엣지 (라우터 함수)
# ============================================================

def should_rewrite_query(state: RAGState) -> str:
    """관련 문서가 부족하면 쿼리 재작성"""
    relevant_docs = state.get("relevant_docs", [])
    rewrite_count = state.get("rewrite_count", 0)

    if len(relevant_docs) == 0 and rewrite_count < MAX_QUERY_REWRITES:
        return "rewrite"
    return "generate"


def should_regenerate(state: RAGState) -> str:
    """환각이 감지되면 재생성"""
    is_grounded = state.get("is_grounded", True)
    generation_count = state.get("generation_count", 0)

    if not is_grounded and generation_count < MAX_GENERATION_RETRIES + 1:
        return "regenerate"
    return "finish"


# ============================================================
# 그래프 빌드
# ============================================================

def build_rag_graph() -> StateGraph:
    """LangGraph RAG 워크플로우 그래프 구축"""
    graph = StateGraph(RAGState)

    # 노드 추가
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade_docs", grade_documents)
    graph.add_node("generate", generate_answer)
    graph.add_node("check_hallucination", check_hallucination_node)
    graph.add_node("finalize", build_final_response)

    # 엣지 연결
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_docs")

    # 조건부: 관련 문서 부족 → 쿼리 재작성 / 충분 → 답변 생성
    graph.add_conditional_edges(
        "grade_docs",
        should_rewrite_query,
        {
            "rewrite": "analyze_query",
            "generate": "generate",
        },
    )

    graph.add_edge("generate", "check_hallucination")

    # 조건부: 환각 감지 → 재생성 / 정상 → 완료
    graph.add_conditional_edges(
        "check_hallucination",
        should_regenerate,
        {
            "regenerate": "generate",
            "finish": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    return graph


# 컴파일된 그래프 (재사용)
_compiled_graph = None


def get_compiled_graph():
    """컴파일된 그래프 싱글턴"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph().compile()
    return _compiled_graph


# ============================================================
# 메인 실행 함수
# ============================================================

def langgraph_rag_query(
    question: str,
    top_k: int = TOP_K,
    researcher: str | None = None,
) -> dict:
    """
    LangGraph 기반 고급 RAG 쿼리 실행

    워크플로우:
    1. 쿼리 분석 및 재작성
    2. 벡터 검색
    3. 문서 관련성 평가 (LLM 기반 그레이딩)
    4. 답변 생성
    5. 환각 검증
    6. (필요 시) 쿼리 재작성 또는 재생성

    Returns:
        {
            "question": str,
            "rewritten_query": str,
            "answer": str,
            "sources": list,
            "context": str,
            "steps": list,          # 실행된 워크플로우 단계
            "is_grounded": bool,
            "rewrite_count": int,
            "generation_count": int,
        }
    """
    compiled = get_compiled_graph()

    initial_state: RAGState = {
        "question": question,
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "sources": [],
        "rewrite_count": 0,
        "generation_count": 0,
        "is_grounded": True,
        "steps": [],
        "researcher": researcher,
        "top_k": top_k,
    }

    final_state = compiled.invoke(initial_state)

    return {
        "question": final_state["question"],
        "rewritten_query": final_state.get("rewritten_query", ""),
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "context": final_state.get("context", ""),
        "steps": final_state.get("steps", []),
        "is_grounded": final_state.get("is_grounded", True),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "generation_count": final_state.get("generation_count", 0),
    }


def langgraph_rag_stream(
    question: str,
    top_k: int = TOP_K,
    researcher: str | None = None,
):
    """
    LangGraph RAG 스트리밍 실행 (각 노드 단계별 결과 반환)

    Yields:
        (node_name, state_update) 튜플
    """
    compiled = get_compiled_graph()

    initial_state: RAGState = {
        "question": question,
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "sources": [],
        "rewrite_count": 0,
        "generation_count": 0,
        "is_grounded": True,
        "steps": [],
        "researcher": researcher,
        "top_k": top_k,
    }

    for event in compiled.stream(initial_state):
        for node_name, state_update in event.items():
            yield node_name, state_update


if __name__ == "__main__":
    import sys
    import json

    q = sys.argv[1] if len(sys.argv) > 1 else "이번 주 주요 업무는?"

    print(f"\n{'='*60}")
    print(f"  LangGraph RAG 워크플로우 실행")
    print(f"  질문: {q}")
    print(f"{'='*60}\n")

    # 스트리밍 모드로 각 단계 확인
    print("[워크플로우 진행 상황]")
    for node_name, update in langgraph_rag_stream(q):
        steps = update.get("steps", [])
        if steps:
            latest_step = steps[-1] if isinstance(steps[-1], str) else steps[-1]
            print(f"  -> {node_name}: {latest_step}")

    # 최종 결과
    print(f"\n{'='*60}")
    result = langgraph_rag_query(q)
    print(f"\n질문: {result['question']}")
    print(f"재작성된 쿼리: {result['rewritten_query']}")
    print(f"\n답변:\n{result['answer']}")
    print(f"\n출처: {len(result['sources'])}건")
    print(f"환각 검증: {'통과' if result['is_grounded'] else '실패'}")
    print(f"쿼리 재작성: {result['rewrite_count']}회")
    print(f"답변 생성: {result['generation_count']}회")
    print(f"\n워크플로우 단계:")
    for step in result["steps"]:
        print(f"  - {step}")
