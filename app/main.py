"""FastAPI 서버: 챗봇 화면, 질문 API, 실행 기록(트레이스).

실행: `python -m uvicorn app.main:app` → http://127.0.0.1:8000 (API 문서는 /docs)

- GET  /                      챗봇 화면
- GET  /health                서버 상태
- GET  /search                검색만 실행 (모델 호출 없음)
- POST /query                 질문 → 검색 → 문맥 → 모델 → 출처 확인 → 답변
- GET  /traces                최근 요청 목록
- GET  /traces/{request_id}   요청 하나의 단계별 기록 + Langfuse 링크
"""
import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import openai
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app.chain import PROMPT, build_chain
from app.config import INPUT_TOKEN_LIMIT, LLM_MODEL, MAX_ANSWER_TOKENS
from app.index import load_resources
from app.retrieve import get_reranker, retrieve
from app.schemas import Mode, QueryRequest, QueryResponse

log = logging.getLogger("uvicorn.error")
STATIC = Path(__file__).parent / "static"

# 화면에서 볼 최근 기록 100개만 메모리에 둔다 (서버를 끄면 사라진다. 영구 기록은 Langfuse)
TRACES: OrderedDict[str, dict] = OrderedDict()
MAX_TRACES = 100


def langfuse_enabled() -> bool:
    """Langfuse 키가 둘 다 있으면 기록을 남긴다."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버가 켜질 때 색인 · 모델 · 체인을 준비하고, 꺼질 때 남은 Langfuse 기록을 보낸다."""
    app.state.res = load_resources()
    get_reranker()  # 리랭커를 미리 불러 둬서 첫 질문이 느리지 않게 한다
    app.state.chain = None
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, max_tokens=MAX_ANSWER_TOKENS, timeout=60, max_retries=1)
        app.state.chain = build_chain(app.state.res, llm)
    else:
        log.warning("OPENAI_API_KEY가 없어 /query를 쓸 수 없습니다. /search는 됩니다.")
    if not langfuse_enabled():
        log.warning("Langfuse 키가 없어 실행 기록을 Langfuse에 남기지 않습니다.")
    yield
    if langfuse_enabled():
        from langfuse import get_client
        get_client().flush()
    app.state.res.qdrant.close()


app = FastAPI(title="금융 RAG 사내 챗봇", lifespan=lifespan)


def chunk_rows(ranked: list[tuple[str, float]]) -> list[dict]:
    """[(chunk_id, 점수)] 목록을 화면에 보여 줄 형태로 바꾼다."""
    by_id = app.state.res.by_id
    return [{"rank": i, "score": round(score, 4), "chunk_id": cid, "book_id": by_id[cid]["book_id"],
             "page": by_id[cid]["page"], "text": by_id[cid]["text"]}
            for i, (cid, score) in enumerate(ranked, start=1)]


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    """서버가 살아 있는지, 모델과 Langfuse를 쓸 수 있는지 알려 준다."""
    return {"status": "ok", "chunks": len(app.state.res.chunks), "llm_ready": app.state.chain is not None,
            "llm_model": LLM_MODEL, "langfuse": langfuse_enabled(),
            "input_token_limit": INPUT_TOKEN_LIMIT, "max_answer_tokens": MAX_ANSWER_TOKENS}


@app.get("/search")
def search(q: str = Query(min_length=1, max_length=1000), mode: Mode = "hybrid_rerank") -> list[dict]:
    """검색 결과만 돌려준다. 세 검색 방식을 눈으로 비교할 때 쓴다."""
    return chunk_rows(retrieve(app.state.res, q, mode))


def save_trace(request_id: str, req: QueryRequest, state: dict, total_ms: int, error: str | None) -> None:
    """요청 하나의 기록(질문, 검색 결과, 최종 문맥, 모델 입출력, 시간, 토큰)을 메모리에 남긴다."""
    sources = state.get("sources", {})
    draft = state.get("draft")
    llm_input = None
    if sources:  # 문맥이 비었으면 모델을 부르지 않았다
        llm_input = [{"role": m.type, "content": m.content}
                     for m in PROMPT.format_messages(context=state["context"], question=req.question)]
    TRACES[request_id] = {
        "request_id": request_id,
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": req.question,
        "retrieval_mode": req.retrieval_mode,
        "status": "error" if error else state.get("status"),
        "total_ms": total_ms,
        "steps": state["steps"],
        "error": error,
        "candidates": chunk_rows(state.get("candidates", [])),
        "context_labels": {label: c["chunk_id"] for label, c in sources.items()},
        # 모델이 실제로 본 문맥. 평가(RAGAS · Judge)가 이 값을 그대로 쓴다
        "contexts": [{"label": label, "chunk_id": c["chunk_id"], "book_id": c["book_id"], "page": c["page"],
                      "text": c["text"]} for label, c in sources.items()],
        "llm_input": llm_input,
        "llm_output": state.get("llm_output"),
        "parsed": draft.model_dump() if draft else None,
        "usage": state.get("usage"),
        "answer": state.get("answer"),
        "cited": state.get("cited", []),
        "langfuse": langfuse_enabled(),
    }
    while len(TRACES) > MAX_TRACES:
        TRACES.popitem(last=False)  # 가장 오래된 기록부터 지운다


@app.post("/query", response_model=QueryResponse,
          responses={502: {"model": QueryResponse, "description": "모델 호출 실패 또는 모델 답 검증 실패"},
                     504: {"model": QueryResponse, "description": "모델 응답 시간 초과"}})
def query(req: QueryRequest):
    """질문에 답하고 출처를 붙여 돌려준다.

    응답 코드
    - 200 처리함 (답변 · 추가 확인 · 보류는 본문 status로 구분)
    - 422 요청 형식이 틀림 (빈 질문, 없는 검색 방식 등)
    - 502 모델 호출이 실패했거나 모델의 답이 검증을 통과하지 못함
    - 503 API 키가 없어 모델을 쓸 수 없음
    - 504 모델 응답 시간 초과
    502 · 504도 본문은 같은 형식(status=error)이라 화면이 똑같이 그릴 수 있다.
    """
    if app.state.chain is None:
        raise HTTPException(503, "생성 모델 설정(OPENAI_API_KEY)이 없습니다.")
    # 요청 ID를 Langfuse trace ID로도 그대로 써서 화면과 Langfuse에서 같은 ID로 찾게 한다
    request_id = uuid4().hex
    callbacks = []
    if langfuse_enabled():
        from langfuse.langchain import CallbackHandler
        callbacks.append(CallbackHandler(trace_context={"trace_id": request_id}))
    config = {"run_name": "query", "callbacks": callbacks,
              "metadata": {"request_id": request_id, "retrieval_mode": req.retrieval_mode}}

    state = {"request_id": request_id, "question": req.question, "mode": req.retrieval_mode, "steps": []}
    t0 = time.perf_counter()
    error, code = None, 200
    try:
        state = app.state.chain.invoke(state, config=config)
        if state["status"] == "error":  # 모델의 답이 검증에서 걸림 (형식 오류, 없는 출처 등)
            code = 502
    except Exception as e:  # 모델 호출 실패 · 시간 초과. "자료가 없어 보류"와는 다른 시스템 오류다
        log.exception("request %s 처리 실패", request_id)
        error = repr(e)
        code = 504 if isinstance(e, (openai.APITimeoutError, TimeoutError)) else 502
    save_trace(request_id, req, state, round((time.perf_counter() - t0) * 1000), error)

    if error:
        body = QueryResponse(request_id=request_id, retrieval_mode=req.retrieval_mode, status="error",
                             answer="처리 중 오류가 발생했습니다.", sources=[])
    else:
        body = QueryResponse(request_id=request_id, retrieval_mode=req.retrieval_mode, status=state["status"],
                             answer=state["answer"], sources=state["cited"])
    return body if code == 200 else JSONResponse(status_code=code, content=body.model_dump())


@app.get("/traces")
def list_traces() -> list[dict]:
    """최근 요청 목록, 최신순."""
    keys = ("request_id", "time", "question", "retrieval_mode", "status", "total_ms")
    return [{k: t[k] for k in keys} for t in reversed(TRACES.values())]


@app.get("/traces/{request_id}")
def get_trace(request_id: str) -> dict:
    """요청 하나의 상세 기록. Langfuse를 쓰면 같은 기록의 Langfuse 주소도 붙인다."""
    trace = TRACES.get(request_id)
    if trace is None:
        raise HTTPException(404, "기록이 없습니다 (서버를 다시 켜면 메모리 기록은 사라집니다).")
    url = None
    if trace["langfuse"]:
        try:
            from langfuse import get_client
            url = get_client().get_trace_url(trace_id=request_id)
        except Exception:  # 링크를 못 만들어도 화면의 기록은 보여 준다
            log.warning("Langfuse 주소를 만들지 못했습니다: %s", request_id)
    return {**trace, "langfuse_url": url}
