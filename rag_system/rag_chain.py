"""
LangChain 기반 RAG 체인 모듈
- LangChain의 ChatOpenAI, PromptTemplate, Retriever를 활용
- 기존 ChromaDB 벡터스토어와 통합
- 구조화된 프롬프트 템플릿 기반 응답 생성
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from embed_store import vector_store
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, TOP_K, SCORE_THRESHOLD


# ============================================================
# LangChain용 Retriever 래퍼
# ============================================================
class ChromaRetriever:
    """기존 ChromaDB VectorStore를 LangChain 체인에서 사용할 수 있도록 래핑"""

    def __init__(self, top_k: int = TOP_K, researcher: str | None = None):
        self.top_k = top_k
        self.researcher = researcher

    def invoke(self, query: str) -> list[dict]:
        """검색 수행 후 결과 반환"""
        where_filter = None
        if self.researcher:
            where_filter = {"researcher": self.researcher}

        query_embedding = vector_store.model.encode([query]).tolist()

        search_params = {
            "query_embeddings": query_embedding,
            "n_results": self.top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            search_params["where"] = where_filter

        results = vector_store.collection.query(**search_params)

        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = 1.0 - distance
                if score < SCORE_THRESHOLD:
                    continue
                formatted.append({
                    "text": results["documents"][0][i],
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i],
                })
        return formatted


# ============================================================
# 프롬프트 템플릿
# ============================================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 연구소 주간업무 보고서 분석 도우미입니다.
주어진 컨텍스트(검색된 보고서 내용)를 기반으로 사용자의 질문에 정확하게 답변하세요.

규칙:
1. 반드시 컨텍스트에 있는 정보만 사용하세요.
2. 컨텍스트에 없는 내용은 "해당 정보가 보고서에 없습니다"라고 답하세요.
3. 출처(파일명, 연구원, 페이지)를 함께 언급하세요.
4. 간결하고 구조화된 답변을 제공하세요.
5. 한국어로 답변하세요."""),
    ("human", """## 검색된 보고서 컨텍스트:
{context}

## 질문:
{question}"""),
])

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 벡터 검색에 더 적합한 형태로 재작성하세요.
한국어 보고서 검색에 최적화된 키워드와 표현을 사용하세요.
재작성된 쿼리만 출력하세요. 설명이나 부가 텍스트는 포함하지 마세요."""),
    ("human", "원본 질문: {question}"),
])

RELEVANCE_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 문서 관련성 판단 전문가입니다.
검색된 문서가 사용자의 질문에 답하는 데 관련이 있는지 판단하세요.
관련이 있으면 "yes", 없으면 "no"만 출력하세요."""),
    ("human", """질문: {question}

문서 내용:
{document}

이 문서가 질문에 답하는 데 관련이 있습니까? (yes/no)"""),
])

HALLUCINATION_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 사실 확인 전문가입니다.
생성된 답변이 제공된 컨텍스트에 기반하여 작성되었는지 확인하세요.
답변이 컨텍스트에 있는 정보에만 기반한 경우 "grounded",
컨텍스트에 없는 정보를 포함한 경우 "hallucinated"만 출력하세요."""),
    ("human", """컨텍스트:
{context}

답변:
{answer}

이 답변은 컨텍스트에 기반합니까? (grounded/hallucinated)"""),
])


# ============================================================
# LangChain RAG 체인 함수
# ============================================================
def format_docs_as_context(docs: list[dict]) -> str:
    """검색 결과를 컨텍스트 문자열로 변환"""
    if not docs:
        return "관련 문서를 찾을 수 없습니다."
    parts = []
    for i, r in enumerate(docs, 1):
        meta = r["metadata"]
        parts.append(
            f"[출처 {i}] 파일: {meta['file_name']} | "
            f"연구원: {meta['researcher']} | "
            f"페이지: {meta['page_num']} | "
            f"유사도: {r['score']:.2f}\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def get_llm() -> ChatOpenAI:
    """LangChain ChatOpenAI 인스턴스 반환"""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
        max_tokens=1500,
    )


def langchain_rag_query(
    question: str,
    top_k: int = TOP_K,
    researcher: str | None = None,
) -> dict:
    """
    LangChain 기반 RAG 쿼리 실행

    Returns:
        {
            "question": str,
            "answer": str,
            "sources": list,
            "context": str,
        }
    """
    retriever = ChromaRetriever(top_k=top_k, researcher=researcher)
    docs = retriever.invoke(question)
    context = format_docs_as_context(docs)

    sources = [
        {
            "file_name": r["metadata"]["file_name"],
            "researcher": r["metadata"]["researcher"],
            "page_num": r["metadata"]["page_num"],
            "score": r["score"],
            "snippet": r["text"][:150],
        }
        for r in docs
    ]

    if not OPENAI_API_KEY:
        return {
            "question": question,
            "answer": f"[LLM 미사용 - API 키 없음]\n\n{context}",
            "sources": sources,
            "context": context,
        }

    try:
        llm = get_llm()
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
    except Exception as e:
        answer = f"[LLM 오류] {e}\n\n아래는 검색된 컨텍스트입니다:\n\n{context}"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context": context,
    }


def rewrite_query(question: str) -> str:
    """LLM을 사용해 검색 쿼리를 재작성"""
    if not OPENAI_API_KEY:
        return question
    try:
        llm = get_llm()
        chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        return chain.invoke({"question": question})
    except Exception:
        return question


def grade_document_relevance(question: str, document: str) -> bool:
    """문서가 질문에 관련 있는지 LLM으로 판단"""
    if not OPENAI_API_KEY:
        return True
    try:
        llm = get_llm()
        chain = RELEVANCE_GRADE_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"question": question, "document": document})
        return result.strip().lower() == "yes"
    except Exception:
        return True


def check_hallucination(context: str, answer: str) -> bool:
    """답변이 컨텍스트에 기반하는지 확인. True이면 grounded."""
    if not OPENAI_API_KEY:
        return True
    try:
        llm = get_llm()
        chain = HALLUCINATION_CHECK_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"context": context, "answer": answer})
        return result.strip().lower() == "grounded"
    except Exception:
        return True


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "이번 주 주요 업무는?"
    result = langchain_rag_query(q)
    print(f"\n질문: {result['question']}\n")
    print(f"답변:\n{result['answer']}\n")
    print(f"출처: {len(result['sources'])}건")
