"""색인 만들기와 불러오기.

- 의미 검색용: 청크를 E5로 벡터화해 Qdrant에 저장한다.
- 키워드 검색용: BM25는 서버가 켜질 때 메모리에서 만든다 (청크 1,992개라 몇 초면 된다).

실행: `python -m app.index` (서버를 끄고 실행. Qdrant 폴더는 한 프로그램만 열 수 있다)
"""
import json
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from kiwipiepy import Kiwi
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import CHUNKS_PATH, COLLECTION, EMBED_MODEL, INDEX_DIR, QDRANT_PATH
from app.ingest import load_chunks

kiwi = Kiwi()
# 키워드로 남길 품사: 명사, 숫자, 영어, 한자, 어근, 동사 · 형용사 어간 (조사 · 어미는 버린다)
KEEP_TAGS = {"NNG", "NNP", "NR", "SL", "SN", "SH", "XR", "VV", "VA"}


def tokenize_ko(text: str) -> list[str]:
    """한국어 문장을 키워드 목록으로 바꾼다. 조사를 떼어 "매출은"과 "매출"이 같은 단어로 잡힌다."""
    return [t.form for t in kiwi.tokenize(text) if t.tag in KEEP_TAGS]


def build() -> None:
    """PDF → 청크 → chunks.json 저장 → 벡터로 바꿔 Qdrant에 새로 저장."""
    chunks = load_chunks()
    INDEX_DIR.mkdir(exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"청크 {len(chunks)}개 → {CHUNKS_PATH}")

    # E5 규칙: 문서 앞에는 "passage: ", 질문 앞에는 "query: "를 붙인다.
    # 길이를 1로 맞춰(normalize) 두면 두 벡터의 곱이 곧 코사인 유사도다.
    vectors = SentenceTransformer(EMBED_MODEL).encode(
        ["passage: " + c["search_text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)

    client = QdrantClient(path=str(QDRANT_PATH))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)  # 예전 청크가 섞이지 않게 처음부터 다시 만든다
    client.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE))
    client.upsert(COLLECTION, points=[
        # 같은 chunk_id는 언제나 같은 ID가 되도록 uuid5를 쓴다
        models.PointStruct(id=str(uuid5(NAMESPACE_URL, c["chunk_id"])), vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)
    ])
    client.close()
    print(f"Qdrant 저장 완료 → {QDRANT_PATH}")


@dataclass
class Resources:
    """서버가 켜질 때 한 번 불러 두고 모든 요청이 같이 쓰는 것."""
    chunks: list[dict]
    by_id: dict[str, dict]       # chunk_id로 청크 원문을 찾는 표
    embedder: SentenceTransformer
    qdrant: QdrantClient
    bm25: BM25Okapi              # BM25 점수의 i번째 = chunks[i]


def load_resources() -> Resources:
    """만들어 둔 색인을 불러오고 BM25를 만든다."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("색인이 없습니다. 먼저 `python -m app.index`를 실행하세요.")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return Resources(
        chunks=chunks,
        by_id={c["chunk_id"]: c for c in chunks},
        embedder=SentenceTransformer(EMBED_MODEL),
        qdrant=QdrantClient(path=str(QDRANT_PATH)),
        bm25=BM25Okapi([tokenize_ko(c["search_text"]) for c in chunks]),
    )


if __name__ == "__main__":
    build()
