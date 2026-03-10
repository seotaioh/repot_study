"""
Step 6: FastAPI 배포 앱
- PDF 업로드 → 자동 파이프라인 (추출 → 청킹 → 임베딩 → 벡터DB)
- 연구원별 데이터 누적 관리
- 시맨틱 검색 / RAG 질의 API
- 파일 관리 (목록, 삭제)
"""
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# rag_system 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import UPLOAD_DIR, PROCESSED_LOG, HOST, PORT
from extract_pdf import extract_and_save
from chunking import chunk_extracted_data
from embed_store import vector_store
from search import search
from rag_query import rag_query


# ============================================================
# 처리 이력 관리
# ============================================================
def load_processed_log() -> dict:
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": []}


def save_processed_log(log: dict):
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def add_to_processed_log(file_info: dict):
    log = load_processed_log()
    log["files"].append(file_info)
    save_processed_log(log)


# ============================================================
# 자동 파이프라인: PDF → 추출 → 청킹 → 임베딩 → 벡터DB
# ============================================================
def process_pdf_pipeline(
    pdf_path: str,
    researcher_name: str,
) -> dict:
    """
    PDF를 전체 파이프라인으로 처리하여 벡터DB에 저장

    Returns:
        처리 결과 요약 딕셔너리
    """
    result = {
        "file_name": Path(pdf_path).name,
        "researcher": researcher_name,
        "status": "processing",
        "steps": {},
    }

    try:
        # Step 1: PDF 텍스트 추출
        extracted = extract_and_save(pdf_path, researcher_name)
        result["steps"]["extract"] = {
            "status": "done",
            "pages": extracted["total_pages"],
            "chars": len(extracted["full_text"]),
        }

        # Step 2: 청킹
        chunks = chunk_extracted_data(extracted)
        result["steps"]["chunk"] = {
            "status": "done",
            "chunk_count": len(chunks),
        }

        # Step 3: 임베딩 + 벡터DB 저장
        stored = vector_store.embed_and_store(chunks)
        result["steps"]["embed_store"] = {
            "status": "done",
            "stored_chunks": stored,
            "total_in_db": vector_store.collection.count(),
        }

        result["status"] = "completed"
        result["processed_at"] = datetime.now().isoformat()

        # 처리 이력 기록
        add_to_processed_log(result)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


# ============================================================
# 기존 PDF 자동 스캔 (서버 시작 시)
# ============================================================
def scan_existing_pdfs():
    """data/uploads 폴더에 있는 미처리 PDF를 자동 처리"""
    if not UPLOAD_DIR.exists():
        return

    log = load_processed_log()
    processed_names = {f["file_name"] for f in log.get("files", [])}

    for pdf_file in UPLOAD_DIR.glob("**/*.pdf"):
        if pdf_file.name not in processed_names:
            # 폴더명을 연구원 이름으로 사용 (uploads/홍길동/report.pdf)
            parts = pdf_file.relative_to(UPLOAD_DIR).parts
            researcher = parts[0] if len(parts) > 1 else "unknown"
            print(f"[자동처리] {pdf_file.name} (연구원: {researcher})")
            process_pdf_pipeline(str(pdf_file), researcher)


# ============================================================
# 프로젝트 루트의 기존 PDF도 초기 처리
# ============================================================
def scan_root_pdfs():
    """프로젝트 루트에 있는 PDF도 처리"""
    base_dir = Path(__file__).parent.parent
    log = load_processed_log()
    processed_names = {f["file_name"] for f in log.get("files", [])}

    for pdf_file in base_dir.glob("*.pdf"):
        if pdf_file.name not in processed_names:
            print(f"[루트PDF 처리] {pdf_file.name}")
            process_pdf_pipeline(str(pdf_file), "unknown")


# ============================================================
# FastAPI 앱
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 기존 PDF 자동 처리"""
    print("=== RAG 시스템 초기화 ===")
    scan_root_pdfs()
    scan_existing_pdfs()
    stats = vector_store.get_stats()
    print(f"DB 현황: 총 {stats['total_chunks']}개 청크, "
          f"{stats['total_files']}개 파일")
    yield
    print("=== RAG 시스템 종료 ===")


app = FastAPI(
    title="연구소 주간업무 RAG 시스템",
    description="PDF 보고서 업로드 → 자동 청킹/임베딩 → 시맨틱 검색 & RAG 질의",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# API 엔드포인트
# ============================================================

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    researcher: str = Form("unknown"),
):
    """
    PDF 업로드 → 자동 파이프라인 처리

    연구원 이름과 함께 PDF를 업로드하면:
    1. 텍스트 추출
    2. 청킹
    3. 임베딩 생성
    4. 벡터DB 저장
    이 자동으로 수행됩니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 업로드 가능합니다.")

    # 연구원별 폴더에 저장
    researcher_dir = UPLOAD_DIR / researcher
    researcher_dir.mkdir(parents=True, exist_ok=True)

    save_path = researcher_dir / file.filename
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 자동 파이프라인 실행
    result = process_pdf_pipeline(str(save_path), researcher)

    return JSONResponse(content=result)


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="검색 질의"),
    top_k: int = Query(5, ge=1, le=20),
    researcher: str = Query(None, description="연구원 필터"),
):
    """시맨틱 검색"""
    results = search(q, top_k=top_k, researcher=researcher)
    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.get("/api/ask")
async def api_ask(
    q: str = Query(..., description="질문"),
    top_k: int = Query(5, ge=1, le=20),
    researcher: str = Query(None, description="연구원 필터"),
):
    """RAG 질의 (검색 + LLM 답변 생성)"""
    result = rag_query(q, top_k=top_k, researcher=researcher)
    return result


@app.get("/api/stats")
async def api_stats():
    """DB 통계 조회"""
    stats = vector_store.get_stats()
    log = load_processed_log()
    stats["processed_history"] = log.get("files", [])
    return stats


@app.get("/api/researchers")
async def api_researchers():
    """등록된 연구원 목록"""
    stats = vector_store.get_stats()
    return {
        "researchers": stats.get("researchers", {}),
    }


@app.delete("/api/file/{file_hash}")
async def delete_file(file_hash: str):
    """특정 파일의 벡터 데이터 삭제"""
    deleted = vector_store.delete_file(file_hash)
    return {
        "deleted_chunks": deleted,
        "file_hash": file_hash,
    }


# ============================================================
# 웹 UI
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """웹 프론트엔드 UI"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>연구소 주간업무 RAG 시스템</h1><p>templates/index.html 을 생성하세요.</p>"


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
