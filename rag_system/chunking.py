"""
Step 2: 텍스트 청킹 모듈
- RecursiveCharacterTextSplitter 기반 청킹
- 페이지/연구원 메타데이터 보존
- 청크별 고유 ID 부여
"""
import json
import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, CHUNKS_DIR


def create_chunks(extracted_data: dict) -> list[dict]:
    """
    추출된 PDF 데이터를 청크로 분할

    Args:
        extracted_data: extract_pdf.py에서 생성된 딕셔너리

    Returns:
        [
            {
                "chunk_id": "고유ID",
                "text": "청크 텍스트",
                "metadata": {
                    "file_name": ...,
                    "file_hash": ...,
                    "researcher": ...,
                    "page_num": ...,
                    "chunk_index": ...,
                    "upload_time": ...,
                }
            },
            ...
        ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    chunks = []
    chunk_index = 0

    for page in extracted_data["pages"]:
        page_text = page["text"] or ""

        # 테이블 텍스트도 포함
        if page.get("tables"):
            page_text += "\n\n" + "\n\n".join(page["tables"])

        if not page_text.strip():
            continue

        splits = splitter.split_text(page_text)

        for split_text in splits:
            # 청크별 고유 ID 생성 (파일해시 + 인덱스)
            chunk_id = hashlib.md5(
                f"{extracted_data['file_hash']}_{chunk_index}".encode()
            ).hexdigest()

            chunk = {
                "chunk_id": chunk_id,
                "text": split_text.strip(),
                "metadata": {
                    "file_name": extracted_data["file_name"],
                    "file_hash": extracted_data["file_hash"],
                    "researcher": extracted_data["researcher"],
                    "page_num": page["page_num"],
                    "chunk_index": chunk_index,
                    "upload_time": extracted_data["upload_time"],
                },
            }
            chunks.append(chunk)
            chunk_index += 1

    return chunks


def save_chunks(chunks: list[dict], file_name: str) -> Path:
    """청크를 JSON으로 저장"""
    safe_name = file_name.replace(".pdf", "").replace(" ", "_")
    out_path = CHUNKS_DIR / f"{safe_name}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return out_path


def chunk_extracted_data(extracted_data: dict) -> list[dict]:
    """청킹 + 저장을 한 번에 수행"""
    chunks = create_chunks(extracted_data)
    saved_path = save_chunks(chunks, extracted_data["file_name"])
    print(f"[청킹완료] {extracted_data['file_name']}")
    print(f"  총 청크 수: {len(chunks)}, 저장: {saved_path}")
    return chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            data = json.load(f)
        chunk_extracted_data(data)
    else:
        print("Usage: python chunking.py <extracted_json_path>")
