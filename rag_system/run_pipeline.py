"""
전체 파이프라인 실행 스크립트
- 기존 PDF 일괄 처리
- CLI에서 개별 PDF 처리
"""
import sys
from pathlib import Path

# rag_system 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_DIR, UPLOAD_DIR
from extract_pdf import extract_and_save
from chunking import chunk_extracted_data
from embed_store import vector_store


def run_single(pdf_path: str, researcher: str = "unknown"):
    """단일 PDF 전체 파이프라인 실행"""
    print(f"\n{'='*60}")
    print(f"  파이프라인 시작: {Path(pdf_path).name}")
    print(f"  연구원: {researcher}")
    print(f"{'='*60}\n")

    # Step 1: 추출
    print("[Step 1/3] PDF 텍스트 추출...")
    extracted = extract_and_save(pdf_path, researcher)

    # Step 2: 청킹
    print("\n[Step 2/3] 텍스트 청킹...")
    chunks = chunk_extracted_data(extracted)

    # Step 3: 임베딩 + 저장
    print("\n[Step 3/3] 임베딩 생성 & 벡터DB 저장...")
    stored = vector_store.embed_and_store(chunks)

    print(f"\n{'='*60}")
    print(f"  파이프라인 완료!")
    print(f"  파일: {extracted['file_name']}")
    print(f"  페이지: {extracted['total_pages']}")
    print(f"  청크: {len(chunks)}개")
    print(f"  저장: {stored}개")
    print(f"  DB 전체 문서: {vector_store.collection.count()}개")
    print(f"{'='*60}\n")

    return stored


def run_all():
    """프로젝트 내 모든 PDF 일괄 처리"""
    pdf_files = []

    # 루트 디렉토리의 PDF
    for f in BASE_DIR.glob("*.pdf"):
        pdf_files.append((str(f), "unknown"))

    # uploads 폴더의 PDF (연구원별)
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.glob("**/*.pdf"):
            parts = f.relative_to(UPLOAD_DIR).parts
            researcher = parts[0] if len(parts) > 1 else "unknown"
            pdf_files.append((str(f), researcher))

    if not pdf_files:
        print("처리할 PDF 파일이 없습니다.")
        return

    print(f"\n총 {len(pdf_files)}개 PDF 파일 발견\n")

    for pdf_path, researcher in pdf_files:
        run_single(pdf_path, researcher)

    # 최종 통계
    stats = vector_store.get_stats()
    print("\n=== 최종 DB 통계 ===")
    print(f"  총 청크: {stats['total_chunks']}개")
    print(f"  총 파일: {stats['total_files']}개")
    print(f"  연구원별:")
    for r, count in stats["researchers"].items():
        print(f"    - {r}: {count}개 청크")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        run_single(pdf, name)
    else:
        run_all()
