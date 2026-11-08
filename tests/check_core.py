"""핵심 규칙 점검. 모델 · 색인 없이 돈다. 실행: `python -m tests.check_core`"""
import os
from types import SimpleNamespace

import httpx
import openai
from fastapi.testclient import TestClient

from app import chain
from app.retrieve import rrf
from app.schemas import LLMAnswer


def check_rrf() -> None:
    """두 검색 결과에 모두 들어 있는 청크가 1등이 된다."""
    fused = rrf([("a", 0.9), ("b", 0.8)], [("b", 5.0), ("c", 4.0)])
    assert fused[0][0] == "b", fused


def check_pack_skips_oversized() -> None:
    """입력 한도를 넘는 청크는 자르지 않고 건너뛰고, 라벨(S1…)은 실제로 넣은 청크에만 붙는다."""
    small = {"chunk_id": "X-p1-c1", "file_name": "X.pdf", "page": 1, "heading": "h", "text": "짧은 문단"}
    huge = {**small, "chunk_id": "X-p1-c2", "text": "매출 " * 5000}
    res = SimpleNamespace(by_id={c["chunk_id"]: c for c in (huge, small)})
    out = chain.pack_context(res, {"question": "q", "candidates": [("X-p1-c2", 1.0), ("X-p1-c1", 0.5)]})
    assert list(out["sources"]) == ["S1"] and out["sources"]["S1"]["chunk_id"] == "X-p1-c1", out["sources"]


def check_finish_rules() -> None:
    """문맥에 없는 라벨을 인용하면 error, 문맥이 비면 abstained, 정상이면 라벨이 문서 · 쪽으로 바뀐다."""
    src = {"S1": {"book_id": "FB01", "file_name": "FB01.pdf", "page": 2, "chunk_id": "FB01-p2-c1"}}
    bad = chain.finish({"sources": src, "draft": LLMAnswer(status="answered", answer="x", source_ids=["S99"])})
    assert bad["status"] == "error"
    empty = chain.finish({"sources": {}, "draft": None})
    assert empty["status"] == "abstained"
    ok = chain.finish({"sources": src, "draft": LLMAnswer(status="answered", answer="x", source_ids=["S1"])})
    assert ok["status"] == "answered" and ok["cited"][0]["page"] == 2


def check_http_codes() -> None:
    """가짜 오류로 응답 코드를 확인한다: 시간 초과 → 504, 모델 호출 실패 → 502, 답 검증 실패 → 502."""
    from app.main import app

    os.environ["LANGFUSE_PUBLIC_KEY"] = ""  # 가짜 시험은 Langfuse에 남기지 않는다

    class Raise:  # invoke하면 주어진 오류를 내는 가짜 체인
        def __init__(self, exc): self.exc = exc
        def invoke(self, state, config=None): raise self.exc

    class BadOutput:  # 검증에서 걸린 답을 돌려주는 가짜 체인
        def invoke(self, state, config=None):
            return {**state, "status": "error", "answer": "문맥에 없는 출처를 인용했습니다: ['S99']", "cited": []}

    app.state.res = SimpleNamespace(by_id={})
    client = TestClient(app)  # with 없이 만들면 서버 시작 단계(색인 불러오기)를 건너뛴다
    cases = [(Raise(openai.APITimeoutError(request=httpx.Request("POST", "http://x"))), 504),
             (Raise(RuntimeError("model down")), 502),
             (BadOutput(), 502)]
    for fake_chain, code in cases:
        app.state.chain = fake_chain
        r = client.post("/query", json={"question": "가짜 오류 시험"})
        assert r.status_code == code and r.json()["status"] == "error" and r.json()["request_id"], (code, r.status_code, r.text)


if __name__ == "__main__":
    check_rrf()
    check_pack_skips_oversized()
    check_finish_rules()
    check_http_codes()
    print("모든 점검 통과")
