"""
Step 3: 임베딩 생성 + ChromaDB 벡터 저장 모듈
- sentence-transformers로 한국어 임베딩
- ChromaDB에 벡터 + 메타데이터 저장
- 중복 방지 (file_hash 기반)
- 데이터 누적 구조 지원
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class VectorStore:
    """ChromaDB 기반 벡터 저장소"""

    def __init__(self):
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[임베딩모델 로딩] {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DB_DIR),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def is_file_already_stored(self, file_hash: str) -> bool:
        """이미 저장된 파일인지 확인 (중복 방지)"""
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1,
        )
        return len(results["ids"]) > 0

    def embed_and_store(self, chunks: list[dict]) -> int:
        """
        청크 리스트를 임베딩하여 ChromaDB에 저장

        Args:
            chunks: chunking.py에서 생성된 청크 리스트

        Returns:
            저장된 청크 수
        """
        if not chunks:
            print("[경고] 저장할 청크가 없습니다.")
            return 0

        # 중복 확인
        file_hash = chunks[0]["metadata"]["file_hash"]
        if self.is_file_already_stored(file_hash):
            print(f"[스킵] 이미 저장된 파일입니다: {chunks[0]['metadata']['file_name']}")
            return 0

        # 텍스트 추출
        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # 배치 임베딩
        print(f"[임베딩 생성 중] {len(texts)}개 청크...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
        ).tolist()

        # ChromaDB에 저장 (배치 단위)
        batch_size = 100
        stored = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            stored += end - i

        file_name = chunks[0]["metadata"]["file_name"]
        researcher = chunks[0]["metadata"]["researcher"]
        print(f"[저장완료] {file_name} (연구원: {researcher})")
        print(f"  저장 청크: {stored}개, 전체 DB 문서수: {self.collection.count()}")

        return stored

    def get_stats(self) -> dict:
        """DB 통계 조회"""
        total = self.collection.count()

        # 연구원별 통계
        all_data = self.collection.get(include=["metadatas"])
        researchers = {}
        files = {}
        for meta in all_data["metadatas"]:
            r = meta.get("researcher", "unknown")
            f = meta.get("file_name", "unknown")
            researchers[r] = researchers.get(r, 0) + 1
            files[f] = files.get(f, 0) + 1

        return {
            "total_chunks": total,
            "total_files": len(files),
            "researchers": researchers,
            "files": files,
        }

    def delete_file(self, file_hash: str) -> int:
        """특정 파일의 벡터를 삭제"""
        results = self.collection.get(where={"file_hash": file_hash})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0


# 싱글턴 인스턴스
vector_store = VectorStore()


if __name__ == "__main__":
    stats = vector_store.get_stats()
    print(f"DB 통계: {stats}")
