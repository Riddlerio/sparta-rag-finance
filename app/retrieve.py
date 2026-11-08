"""검색 3방식. 뒤 방식은 앞 방식에 단계를 하나씩 더한 것이다.

- naive          : 의미 검색(E5)만 → 상위 10개
- hybrid         : 의미 검색 20개 + 키워드 검색(BM25) 20개 → 순위로 합치기(RRF) → 상위 10개
- hybrid_rerank  : hybrid 후보 20개 → 리랭커가 질문과 함께 다시 읽고 줄 세우기 → 상위 10개

모든 함수는 [(chunk_id, 점수), ...]를 점수 높은 순으로 돌려준다.
"""
from functools import cache

from sentence_transformers import CrossEncoder

from app.config import CANDIDATES, COLLECTION, RERANK_MODEL, RRF_K, TOP_K
from app.index import Resources, tokenize_ko


def dense_search(res: Resources, question: str, n: int) -> list[tuple[str, float]]:
    """의미 검색: 질문 벡터와 코사인 유사도가 높은 청크 n개."""
    vec = res.embedder.encode("query: " + question, normalize_embeddings=True)
    points = res.qdrant.query_points(COLLECTION, query=vec.tolist(), limit=n).points
    return [(p.payload["chunk_id"], p.score) for p in points]


def bm25_search(res: Resources, question: str, n: int) -> list[tuple[str, float]]:
    """키워드 검색: BM25 점수 상위 n개. 겹치는 단어가 하나도 없는 청크(0점)는 뺀다."""
    scores = res.bm25.get_scores(tokenize_ko(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [(res.chunks[i]["chunk_id"], float(scores[i])) for i in top if scores[i] > 0]


def rrf(*rankings: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """여러 검색 결과를 순위로 합친다. 각 목록에서 r등이면 1/(60+r)점을 더한다.

    점수가 아니라 순위만 쓰므로, 척도가 다른 코사인 점수와 BM25 점수를 그대로 합칠 수 있다.
    """
    total: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _score) in enumerate(ranking, start=1):
            total[cid] = total.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(total.items(), key=lambda x: x[1], reverse=True)


@cache
def get_reranker() -> CrossEncoder:
    """리랭커 모델(약 2GB)은 처음 한 번만 불러온다."""
    return CrossEncoder(RERANK_MODEL)


def rerank(res: Resources, question: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """후보마다 (질문, 청크)를 리랭커에 같이 넣어 새 점수를 받고, 그 점수로 다시 줄 세운다."""
    ids = [cid for cid, _ in candidates]
    scores = get_reranker().predict([(question, res.by_id[cid]["search_text"]) for cid in ids])
    return sorted(zip(ids, map(float, scores)), key=lambda x: x[1], reverse=True)


def hybrid_candidates(res: Resources, question: str) -> list[tuple[str, float]]:
    """의미 검색 20개와 키워드 검색 20개를 RRF로 합친 목록."""
    return rrf(dense_search(res, question, CANDIDATES), bm25_search(res, question, CANDIDATES))


def naive_rag(res: Resources, question: str) -> list[tuple[str, float]]:
    return dense_search(res, question, TOP_K)


def hybrid_rag(res: Resources, question: str) -> list[tuple[str, float]]:
    return hybrid_candidates(res, question)[:TOP_K]


def hybrid_rerank_rag(res: Resources, question: str) -> list[tuple[str, float]]:
    return rerank(res, question, hybrid_candidates(res, question)[:CANDIDATES])[:TOP_K]


PIPELINES = {"naive": naive_rag, "hybrid": hybrid_rag, "hybrid_rerank": hybrid_rerank_rag}


def retrieve(res: Resources, question: str, mode: str) -> list[tuple[str, float]]:
    """mode 이름("naive", "hybrid", "hybrid_rerank")에 맞는 검색을 실행한다."""
    return PIPELINES[mode](res, question)
