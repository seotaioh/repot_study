"""
Step 1: PDF 텍스트 추출 모듈
- pdfplumber를 사용하여 PDF에서 텍스트 추출
- 페이지별 메타데이터 보존
- 테이블 데이터도 함께 추출
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime

import pdfplumber

from config import EXTRACTED_DIR


def get_file_hash(file_path: str) -> str:
    """파일의 MD5 해시를 계산하여 중복 처리 방지"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_from_pdf(
    pdf_path: str,
    researcher_name: str = "unknown",
) -> dict:
    """
    PDF에서 텍스트를 추출하고 메타데이터와 함께 반환

    Args:
        pdf_path: PDF 파일 경로
        researcher_name: 업로드한 연구원 이름

    Returns:
        {
            "file_name": ...,
            "file_hash": ...,
            "researcher": ...,
            "upload_time": ...,
            "total_pages": ...,
            "pages": [{"page_num": 1, "text": ..., "tables": [...]}, ...]
        }
    """
    pdf_path = Path(pdf_path)
    file_hash = get_file_hash(str(pdf_path))

    result = {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "file_hash": file_hash,
        "researcher": researcher_name,
        "upload_time": datetime.now().isoformat(),
        "total_pages": 0,
        "pages": [],
        "full_text": "",
    }

    all_text_parts = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        result["total_pages"] = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            tables = page.extract_tables() or []

            # 테이블 데이터를 텍스트로 변환
            table_texts = []
            for table in tables:
                rows = []
                for row in table:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
                table_texts.append("\n".join(rows))

            page_data = {
                "page_num": i + 1,
                "text": page_text,
                "tables": table_texts,
            }
            result["pages"].append(page_data)

            # 전체 텍스트 합산
            combined = page_text
            if table_texts:
                combined += "\n[표]\n" + "\n\n".join(table_texts)
            all_text_parts.append(combined)

    result["full_text"] = "\n\n--- 페이지 구분 ---\n\n".join(all_text_parts)

    return result


def save_extracted(data: dict) -> Path:
    """추출 결과를 JSON으로 저장"""
    safe_name = data["file_name"].replace(".pdf", "").replace(" ", "_")
    out_path = EXTRACTED_DIR / f"{safe_name}_{data['file_hash'][:8]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def extract_and_save(pdf_path: str, researcher_name: str = "unknown") -> dict:
    """추출 + 저장을 한 번에 수행"""
    data = extract_text_from_pdf(pdf_path, researcher_name)
    saved_path = save_extracted(data)
    print(f"[추출완료] {data['file_name']} → {saved_path}")
    print(f"  연구원: {data['researcher']}, 페이지: {data['total_pages']}, "
          f"글자수: {len(data['full_text'])}")
    return data


# 단독 실행 테스트
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        extract_and_save(pdf, name)
    else:
        print("Usage: python extract_pdf.py <pdf_path> [researcher_name]")
