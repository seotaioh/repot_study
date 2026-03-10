"""
Step 5: RAG 쿼리 파이프라인
- 검색된 문서(컨텍스트) + 사용자 질문 → LLM 응답 생성
- OpenAI API 기반 (API 키 없으면 컨텍스트만 반환)
"""
from openai import OpenAI

from search import search
from config import OPENAI_API_KEY, LLM_MODEL, TOP_K


SYSTEM_PROMPT = """당신은 연구소 주간업무 보고서 분석 도우미입니다.
주어진 컨텍스트(검색된 보고서 내용)를 기반으로 사용자의 질문에 정확하게 답변하세요.

규칙:
1. 반드시 컨텍스트에 있는 정보만 사용하세요.
2. 컨텍스트에 없는 내용은 "해당 정보가 보고서에 없습니다"라고 답하세요.
3. 출처(파일명, 연구원, 페이지)를 함께 언급하세요.
4. 간결하고 구조화된 답변을 제공하세요.
"""


def build_context(results: list[dict]) -> str:
    """검색 결과를 LLM 컨텍스트 문자열로 변환"""
    if not results:
        return "관련 문서를 찾을 수 없습니다."

    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        parts.append(
            f"[출처 {i}] 파일: {meta['file_name']} | "
            f"연구원: {meta['researcher']} | "
            f"페이지: {meta['page_num']} | "
            f"유사도: {r['score']:.2f}\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def rag_query(
    question: str,
    top_k: int = TOP_K,
    researcher: str | None = None,
    use_llm: bool = True,
) -> dict:
    """
    RAG 파이프라인 실행: 검색 → 컨텍스트 구성 → LLM 응답

    Args:
        question: 사용자 자연어 질문
        top_k: 검색 결과 수
        researcher: 특정 연구원 필터
        use_llm: LLM 응답 생성 여부

    Returns:
        {
            "question": ...,
            "answer": ...,
            "sources": [...],
            "context": ...,
        }
    """
    # 1) 시맨틱 검색
    results = search(question, top_k=top_k, researcher=researcher)

    # 2) 컨텍스트 구성
    context = build_context(results)

    # 3) 소스 정보
    sources = [
        {
            "file_name": r["metadata"]["file_name"],
            "researcher": r["metadata"]["researcher"],
            "page_num": r["metadata"]["page_num"],
            "score": r["score"],
            "snippet": r["text"][:150],
        }
        for r in results
    ]

    # 4) LLM 응답 생성
    answer = ""
    if use_llm and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            user_message = (
                f"## 검색된 보고서 컨텍스트:\n{context}\n\n"
                f"## 질문:\n{question}"
            )
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"[LLM 오류] {e}\n\n아래는 검색된 컨텍스트입니다:\n\n{context}"
    else:
        answer = (
            "[LLM 미사용 - 검색된 컨텍스트 직접 반환]\n\n"
            f"{context}"
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context": context,
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "이번 주 주요 업무는?"
    result = rag_query(q)
    print(f"\n질문: {result['question']}\n")
    print(f"답변:\n{result['answer']}\n")
    print(f"출처: {len(result['sources'])}건")
