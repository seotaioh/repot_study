"""
Step 4: 시맨틱 검색 모듈
- 사용자 질의를 임베딩하여 ChromaDB에서 유사 문서 검색
- 연구원별/파일별 필터링 지원
- 검색 결과에 메타데이터 포함
"""
from embed_store import vector_store
from config import TOP_K, SCORE_THRESHOLD


def search(
    query: str,
    top_k: int = TOP_K,
    researcher: str | None = None,
    file_name: str | None = None,
) -> list[dict]:
    """
    시맨틱 검색 수행

    Args:
        query: 검색 질의 (자연어)
        top_k: 반환할 최대 결과 수
        researcher: 특정 연구원 필터 (선택)
        file_name: 특정 파일 필터 (선택)

    Returns:
        [
            {
                "text": "매칭된 청크 텍스트",
                "score": 0.85,
                "metadata": {...}
            },
            ...
        ]
    """
    # 필터 조건 구성
    where_filter = None
    conditions = []
    if researcher:
        conditions.append({"researcher": researcher})
    if file_name:
        conditions.append({"file_name": file_name})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    # 질의 임베딩 + 검색
    query_embedding = vector_store.model.encode([query]).tolist()

    search_params = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        search_params["where"] = where_filter

    results = vector_store.collection.query(**search_params)

    # 결과 정리
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # cosine distance → similarity (1 - distance)
            score = 1.0 - distance

            if score < SCORE_THRESHOLD:
                continue

            formatted.append({
                "text": results["documents"][0][i],
                "score": round(score, 4),
                "metadata": results["metadatas"][0][i],
            })

    return formatted


def search_pretty(
    query: str,
    top_k: int = TOP_K,
    researcher: str | None = None,
) -> str:
    """검색 결과를 보기 좋게 포맷팅"""
    results = search(query, top_k, researcher)

    if not results:
        return f"'{query}'에 대한 검색 결과가 없습니다."

    lines = [f"== 검색: '{query}' (결과 {len(results)}건) ==\n"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        lines.append(f"[{i}] 유사도: {r['score']:.4f}")
        lines.append(f"    파일: {meta['file_name']}")
        lines.append(f"    연구원: {meta['researcher']} | 페이지: {meta['page_num']}")
        lines.append(f"    내용: {r['text'][:200]}...")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    query = sys.argv[1] if len(sys.argv) > 1 else "주간 업무 보고"
    print(search_pretty(query))
