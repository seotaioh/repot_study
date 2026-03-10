"""
RAG 시스템 설정 파일
연구소 주간업무 보고서 검색 시스템
LangChain + LangGraph 기반 고급 RAG 파이프라인
"""
import os
from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path(__file__).parent.parent
RAG_DIR = Path(__file__).parent

# PDF 업로드 폴더 (연구원이 PDF를 넣으면 자동 처리)
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 추출된 텍스트 저장
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# 청크 저장
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB 저장 경로
CHROMA_DB_DIR = BASE_DIR / "db" / "chroma"
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# 처리 완료 추적 파일
PROCESSED_LOG = BASE_DIR / "data" / "processed_files.json"

# ============================================================
# 청킹 설정
# ============================================================
CHUNK_SIZE = 500          # 청크 당 글자 수
CHUNK_OVERLAP = 100       # 청크 간 겹침 글자 수
SEPARATORS = ["\n\n", "\n", ".", "。", " ", ""]

# ============================================================
# 임베딩 모델 설정
# ============================================================
# 한국어 지원 다국어 모델
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# 대안: "jhgan/ko-sroberta-multitask" (한국어 전용, 더 정확)

# ============================================================
# ChromaDB 컬렉션 설정
# ============================================================
COLLECTION_NAME = "weekly_reports"

# ============================================================
# LLM 설정 (RAG 응답 생성용)
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"  # 비용 효율적, "gpt-4o"로 변경 가능
LLM_TEMPERATURE = 0.2     # 응답 생성 온도

# ============================================================
# 검색 설정
# ============================================================
TOP_K = 5                 # 검색 시 반환할 상위 문서 수
SCORE_THRESHOLD = 0.3     # 유사도 임계값 (낮을수록 엄격)

# ============================================================
# LangGraph RAG 설정
# ============================================================
# 쿼리 재작성 최대 횟수
MAX_QUERY_REWRITES = 2

# 문서 관련성 판단 임계 점수 (LLM 기반 그레이딩)
RELEVANCE_THRESHOLD = 0.5

# 최종 답변 환각 검증 여부
ENABLE_HALLUCINATION_CHECK = True

# 답변 품질 검증 후 재생성 최대 횟수
MAX_GENERATION_RETRIES = 1

# ============================================================
# 서버 설정
# ============================================================
HOST = "0.0.0.0"
PORT = 8000
