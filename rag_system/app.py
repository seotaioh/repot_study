"""
Step 6: FastAPI 배포 앱
- PDF 업로드 → 자동 파이프라인 (추출 → 청킹 → 임베딩 → 벡터DB)
- 연구원별 데이터 누적 관리
- 시맨틱 검색 / RAG 질의 API (기본 + LangChain + LangGraph)
- LangGraph 기반 고급 RAG: 쿼리 재작성, 문서 그레이딩, 환각 검증
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
from rag_chain import langchain_rag_query
from rag_graph import langgraph_rag_query, langgraph_rag_stream
from auth import (
    get_user, get_all_users, add_user, update_user, delete_user,
    check_view_permission, get_viewable_researchers, can_delete_file,
    load_users_config,
)


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
    description="PDF 보고서 업로드 → 자동 청킹/임베딩 → 시맨틱 검색 & LangChain/LangGraph RAG 질의",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# API 엔드포인트
# ============================================================

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    researcher: str = Form("unknown"),
    current_user: str = Form("unknown"),
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

    # 로그인된 사용자 확인
    user = get_user(current_user)
    if not user:
        raise HTTPException(403, f"등록되지 않은 사용자입니다: {current_user}")

    # 업로드 시 연구원 이름은 로그인된 사용자 이름으로 자동 설정
    researcher = current_user

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
    current_user: str = Query("unknown", description="현재 로그인 사용자"),
):
    """시맨틱 검색 (권한 기반 필터링)"""
    user = get_user(current_user)
    if not user:
        raise HTTPException(403, f"등록되지 않은 사용자입니다: {current_user}")

    # 권한에 따른 열람 범위 제한
    viewable = get_viewable_researchers(current_user)

    # 특정 연구원 필터가 지정된 경우 권한 확인
    if researcher:
        if viewable is not None and researcher not in viewable:
            raise HTTPException(403, f"{researcher}의 보고서를 열람할 권한이 없습니다.")
        results = search(q, top_k=top_k, researcher=researcher)
    elif viewable is not None:
        # 전체 검색이지만 열람 범위 제한이 있는 경우: 허용된 연구원별로 검색 후 통합
        all_results = []
        for name in viewable:
            partial = search(q, top_k=top_k, researcher=name)
            all_results.extend(partial)
        # 유사도 점수로 정렬 후 top_k 제한
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = all_results[:top_k]
    else:
        # 전체 열람 가능
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
    current_user: str = Query("unknown", description="현재 로그인 사용자"),
):
    """RAG 질의 - 기본 모드 (권한 기반 필터링)"""
    user = get_user(current_user)
    if not user:
        raise HTTPException(403, f"등록되지 않은 사용자입니다: {current_user}")

    # 권한 기반 연구원 필터 적용
    effective_researcher = _apply_view_filter(current_user, researcher)
    result = rag_query(q, top_k=top_k, researcher=effective_researcher)
    return result


@app.get("/api/ask/langchain")
async def api_ask_langchain(
    q: str = Query(..., description="질문"),
    top_k: int = Query(5, ge=1, le=20),
    researcher: str = Query(None, description="연구원 필터"),
    current_user: str = Query("unknown", description="현재 로그인 사용자"),
):
    """LangChain 기반 RAG 질의 (권한 기반 필터링)"""
    user = get_user(current_user)
    if not user:
        raise HTTPException(403, f"등록되지 않은 사용자입니다: {current_user}")

    effective_researcher = _apply_view_filter(current_user, researcher)
    result = langchain_rag_query(q, top_k=top_k, researcher=effective_researcher)
    return result


@app.get("/api/ask/langgraph")
async def api_ask_langgraph(
    q: str = Query(..., description="질문"),
    top_k: int = Query(5, ge=1, le=20),
    researcher: str = Query(None, description="연구원 필터"),
    current_user: str = Query("unknown", description="현재 로그인 사용자"),
):
    """LangGraph 기반 고급 RAG 질의 (권한 기반 필터링)"""
    user = get_user(current_user)
    if not user:
        raise HTTPException(403, f"등록되지 않은 사용자입니다: {current_user}")

    effective_researcher = _apply_view_filter(current_user, researcher)
    result = langgraph_rag_query(q, top_k=top_k, researcher=effective_researcher)
    return result


def _apply_view_filter(current_user: str, requested_researcher: str | None) -> str | None:
    """
    권한에 따라 검색할 연구원 필터를 결정
    - 전체 열람 가능 사용자: 요청된 필터 그대로 사용
    - 제한된 사용자: 본인 이름으로 강제 필터링
    """
    viewable = get_viewable_researchers(current_user)
    if viewable is None:
        # 전체 열람 가능
        return requested_researcher

    if requested_researcher:
        if requested_researcher not in viewable:
            return current_user  # 권한 없는 연구원 → 본인으로 대체
        return requested_researcher

    # 열람 가능 대상이 1명(본인)이면 자동 필터링
    if len(viewable) == 1:
        return viewable[0]

    # 여러 명 열람 가능하지만 전체는 아닌 경우 → 필터 없이 (검색에서 필터링)
    return None


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
async def delete_file(
    file_hash: str,
    current_user: str = Query("unknown"),
):
    """특정 파일의 벡터 데이터 삭제 (삭제 권한 필요)"""
    user = get_user(current_user)
    if not user:
        raise HTTPException(403, "등록되지 않은 사용자입니다.")
    if not can_delete_file(current_user):
        raise HTTPException(403, "파일 삭제 권한이 없습니다.")

    deleted = vector_store.delete_file(file_hash)
    return {
        "deleted_chunks": deleted,
        "file_hash": file_hash,
    }


# ============================================================
# 사용자 관리 API
# ============================================================

@app.get("/api/users")
async def api_get_users():
    """전체 사용자 목록 조회"""
    users = get_all_users()
    result = {}
    for name, info in users.items():
        result[name] = {
            "role": info.get("role", ""),
            "department": info.get("department", ""),
            "view_scope": info.get("view_scope", "self"),
            "can_delete": info.get("can_delete", False),
        }
    return {"users": result}


@app.get("/api/users/{name}")
async def api_get_user(name: str):
    """특정 사용자 정보 조회"""
    user = get_user(name)
    if not user:
        raise HTTPException(404, f"사용자를 찾을 수 없습니다: {name}")
    return {"user": user}


@app.post("/api/users")
async def api_add_user(
    name: str = Form(...),
    role: str = Form(...),
    department: str = Form("연구소"),
    view_scope: str = Form("self"),
    can_delete: bool = Form(False),
    admin_user: str = Form(...),
):
    """
    새 사용자 추가 (소장/차장만 가능)

    view_scope 설정:
    - "all": 전체 열람
    - "self": 본인만
    - 쉼표 구분 이름 목록: 지정 인원 열람 (예: "이치형,이재원")
    """
    admin = get_user(admin_user)
    if not admin:
        raise HTTPException(403, "등록되지 않은 사용자입니다.")

    config = load_users_config()
    admin_level = config.get("roles_hierarchy", {}).get(admin.get("role", ""), 0)
    if admin_level < 4:
        raise HTTPException(403, "사용자 추가 권한이 없습니다. (차장 이상만 가능)")

    # view_scope 파싱: 쉼표 구분 목록이면 리스트로 변환
    parsed_scope = view_scope
    if view_scope not in ("all", "self") and "," in view_scope:
        parsed_scope = [s.strip() for s in view_scope.split(",") if s.strip()]

    result = add_user(name, role, department, parsed_scope, can_delete)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


@app.put("/api/users/{name}")
async def api_update_user(
    name: str,
    role: str = Form(None),
    department: str = Form(None),
    view_scope: str = Form(None),
    can_delete: bool = Form(None),
    admin_user: str = Form(...),
):
    """사용자 정보 수정 (소장/차장만 가능)"""
    admin = get_user(admin_user)
    if not admin:
        raise HTTPException(403, "등록되지 않은 사용자입니다.")

    config = load_users_config()
    admin_level = config.get("roles_hierarchy", {}).get(admin.get("role", ""), 0)
    if admin_level < 4:
        raise HTTPException(403, "사용자 수정 권한이 없습니다. (차장 이상만 가능)")

    updates = {}
    if role is not None:
        updates["role"] = role
    if department is not None:
        updates["department"] = department
    if view_scope is not None:
        if view_scope not in ("all", "self") and "," in view_scope:
            updates["view_scope"] = [s.strip() for s in view_scope.split(",") if s.strip()]
        else:
            updates["view_scope"] = view_scope
    if can_delete is not None:
        updates["can_delete"] = can_delete

    result = update_user(name, updates)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


@app.delete("/api/users/{name}")
async def api_delete_user(
    name: str,
    admin_user: str = Query(...),
):
    """사용자 삭제 (소장만 가능)"""
    admin = get_user(admin_user)
    if not admin:
        raise HTTPException(403, "등록되지 않은 사용자입니다.")

    config = load_users_config()
    admin_level = config.get("roles_hierarchy", {}).get(admin.get("role", ""), 0)
    if admin_level < 5:
        raise HTTPException(403, "사용자 삭제 권한이 없습니다. (소장만 가능)")

    result = delete_user(name)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


@app.get("/api/users/{name}/permissions")
async def api_get_permissions(name: str):
    """사용자 권한 정보 상세 조회"""
    user = get_user(name)
    if not user:
        raise HTTPException(404, f"사용자를 찾을 수 없습니다: {name}")

    viewable = get_viewable_researchers(name)
    return {
        "user": name,
        "role": user.get("role", ""),
        "view_scope": user.get("view_scope", "self"),
        "viewable_researchers": viewable,
        "can_view_all": viewable is None,
        "can_delete": user.get("can_delete", False),
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
