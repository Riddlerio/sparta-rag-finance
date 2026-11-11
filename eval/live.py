"""검색 평가지표를 터미널 표로 바로 본다. 코드를 고칠 때마다 다시 재서 이전 결과와 비교한다.

"금융 RAG 코드 지도"의 ③ 평가 흐름에 붙는 파일이다. run_eval이 서버로 답을 모아 채점한다면,
이 파일은 서버 없이 검색만 다시 돌려 ①(ingest · index)과 ②의 retrieve를 고친 효과를 바로 잰다.
    ① ingest(청크) → index(색인) → ② retrieve(검색 3방식) → ③ 평가(이 파일)

지금 다시 재는 지표 (검색만 하므로 OpenAI 키가 필요 없다)
  - Hit@10 · Recall@10 (문서)   상위 10개에 정답 근거 문서가 하나라도 있나 / 정답 문서 중 몇 개를 찾았나
  - 정답 청크 Hit@1 · Hit@10     정답 파일 quote가 든 문단(청크)이 1등 / 10개 안에 있나
  - 근거 묶음 Recall@10          근거 묶음(evidence_groups) 중 몇 개의 정답 청크를 찾았나
  - MRR(청크)                    첫 정답 청크 순위의 역수 평균 (1등 = 1, 2등 = 0.5 …)
  - 검색 시간                    질문 하나 검색에 걸린 평균 시간
  정답이 없는 질문(보류가 정답)은 검색 지표에서 빼고, 몇 문항으로 계산했는지 n으로 보여 준다.

함께 보여 주는 저장된 답변 지표 (eval/results/summary_<묶음>.json, 다시 재려면 키 + run_eval)
  - RAGAS Faithfulness, LLM Judge 사실 · 조건 · 근거 · 판단, 평균 응답 시간, 오류율, 토큰

실행 (서버를 끄고: Qdrant 폴더는 한 프로그램만 열 수 있다)
  python -m eval.live                    한 번 재고 표 출력
  python -m eval.live --watch            app/ · eval/ 파일을 저장할 때마다 다시 잰다
                                         (ingest · index · config가 바뀌면 색인부터 다시 만든다)
  python -m eval.live --modes naive,hybrid   리랭커를 빼고 빠르게 (리랭커는 CPU에서 질문당 약 5초)
  python -m eval.live --set <묶음>       다른 질문 묶음 (기본 development 30문항, data/evaluation/<묶음>.json)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

# 모델을 불러올 때 나오는 진행 막대 · 경고를 숨겨 표만 보이게 한다
for key, value in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                   "TRANSFORMERS_VERBOSITY": "error", "TQDM_DISABLE": "1"}.items():
    os.environ.setdefault(key, value)

from app.config import EVAL_DIR, ROOT, TOP_K  # noqa: E402

OUT = ROOT / "eval" / "results"
MODES = ["naive", "hybrid", "hybrid_rerank"]
LABEL = {"naive": "naive", "hybrid": "hybrid", "hybrid_rerank": "hybrid+rerank"}
console = Console()


# ── 1. 재기: 질문마다 검색하고 순위로 지표 계산 ─────────────────────────────

def measure(split: str, modes: list[str]) -> dict:
    """질문 묶음을 각 방식으로 검색하고 방식별 지표를 돌려준다. 정답 파일은 검색이 끝난 뒤에만 연다."""
    from app.index import load_resources
    from app.retrieve import PIPELINES
    from eval.run_eval import chunk_rank_metrics, gold_books

    res = load_resources()
    questions = json.loads((EVAL_DIR / f"{split}_questions.json").read_text(encoding="utf-8"))
    if "hybrid_rerank" in modes:
        PIPELINES["hybrid_rerank"](res, "준비")  # 리랭커를 미리 불러 첫 질문 시간이 부풀지 않게 한다

    # 1-1. 검색 (질문만 사용)
    ranked, ms = {m: {} for m in modes}, {m: [] for m in modes}
    for i, q in enumerate(questions, start=1):
        for m in modes:
            t0 = time.perf_counter()
            ranked[m][q["id"]] = [cid for cid, _ in PIPELINES[m](res, q["question"])]
            ms[m].append((time.perf_counter() - t0) * 1000)
        console.print(f"  검색 {i}/{len(questions)}", end="\r")

    # 1-2. 채점 (여기서 처음 정답 파일을 연다)
    answers = {a["id"]: a for a in json.loads((EVAL_DIR / f"{split}.json").read_text(encoding="utf-8"))}
    by_page = {}
    for c in res.chunks:
        by_page.setdefault((c["book_id"], c["page"]), []).append(c)

    out = {"split": split, "chunks": len(res.chunks), "questions": len(questions), "k": TOP_K,
           "time": datetime.now().strftime("%H:%M:%S"), "modes": {}}
    for m in modes:
        scored = [qid for qid in ranked[m] if gold_books(answers[qid])]
        hits, recalls = [], []
        for qid in scored:
            gold = gold_books(answers[qid])
            books = {res.by_id[c]["book_id"] for c in ranked[m][qid]}
            hits.append(int(bool(gold & books)))
            recalls.append(len(gold & books) / len(gold))
        chunk = chunk_rank_metrics(ranked[m], answers, by_page)
        out["modes"][m] = {"hit": mean(hits), "recall": mean(recalls), "doc_n": len(scored),
                           "chunk_hit1": chunk["chunk_hit@1"], "chunk_hit10": chunk["chunk_hit@10"],
                           "group_recall": chunk["group_recall@10"], "mrr": chunk["chunk_mrr"], "chunk_n": chunk["n"],
                           "search_ms": mean(ms[m])}
    return out


# ── 2. 보여 주기: 터미널 표 (이전 실행과의 차이를 ▲▼로) ─────────────────────

SEARCH_ROWS = [  # (키, 이름, 형식, 클수록 좋은가)
    ("hit", "Hit@10 (문서)", "{:.3f}", True),
    ("recall", "Recall@10 (문서)", "{:.3f}", True),
    ("chunk_hit1", "정답 청크 Hit@1", "{:.3f}", True),
    ("chunk_hit10", "정답 청크 Hit@10", "{:.3f}", True),
    ("group_recall", "근거 묶음 Recall@10", "{:.3f}", True),
    ("mrr", "MRR (청크)", "{:.3f}", True),
    ("search_ms", "검색 시간 (평균)", "{:,.0f}ms", False),
]


def delta(now: float, before: float | None, higher_better: bool) -> str:
    """이전 실행보다 좋아지면 초록 ▲, 나빠지면 빨강 ▼. 시간은 작을수록 좋다."""
    if before is None or abs(now - before) < 1e-9:
        return ""
    if not higher_better and abs(now - before) < max(50, before * 0.3):
        return ""  # 시간은 실행마다 흔들리므로 50ms · 30% 미만 차이는 표시하지 않는다
    better = (now > before) == higher_better
    diff = f"{now - before:+.0f}ms" if not higher_better else f"{now - before:+.3f}"
    return f" [{'green' if better else 'red'}]{'▲' if now > before else '▼'}{diff}[/]"


def search_table(result: dict, previous: dict | None) -> Table:
    modes = list(result["modes"])
    t = Table(title=f"검색 지표 · {result['split']} {result['questions']}문항 · 청크 {result['chunks']:,}개 · k={result['k']} · {result['time']}",
              title_justify="left", header_style="bold")
    t.add_column("지표")
    for m in modes:
        t.add_column(LABEL[m], justify="right")
    for key, name, fmt, up in SEARCH_ROWS:
        cells = []
        best = (max if up else min)(result["modes"][m][key] for m in modes)
        for m in modes:
            v = result["modes"][m][key]
            before = (previous or {}).get("modes", {}).get(m, {}).get(key)
            text = fmt.format(v)
            cells.append((f"[bold]{text}[/]" if v == best and len(modes) > 1 else text) + delta(v, before, up))
        t.add_row(name, *cells)
    first = result["modes"][modes[0]]
    t.caption = f"문서 지표 n={first['doc_n']} · 청크 지표 n={first['chunk_n']} (정답 근거가 없는 질문 제외) · 굵게 = 세 방식 중 최고"
    t.caption_justify = "left"
    return t


ANSWER_ROWS = [  # (이름, summary에서 값 꺼내기)
    ("Faithfulness (RAGAS)", lambda s: s["faithfulness"]["mean"]),
    ("Judge 사실", lambda s: s["judge_fact"]["mean"]),
    ("Judge 조건", lambda s: s["judge_condition"]["mean"]),
    ("Judge 근거", lambda s: s["judge_evidence"]["mean"]),
    ("Judge 판단", lambda s: s["judge_judgment"]["mean"]),
    ("평균 응답 시간", lambda s: f"{s['ops']['avg_response_ms']:,}ms"),
    ("오류율", lambda s: f"{s['ops']['error_rate_pct']:.1f}%"),
    ("입력 / 출력 토큰", lambda s: f"{s['ops']['input_tokens']:,} / {s['ops']['output_tokens']:,}"),
]


def answer_table(split: str) -> Table | None:
    """저장된 답변 채점 결과. 키가 없어도 볼 수 있지만, 코드를 고쳐도 이 값은 바뀌지 않는다."""
    path = OUT / f"summary_{split}.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    modes = [m for m in MODES if m in summary["modes"]]
    t = Table(title=f"답변 지표 · 저장된 채점 ({path.name}, 생성 {summary.get('model')})", title_justify="left", header_style="bold")
    t.add_column("지표")
    for m in modes:
        t.add_column(LABEL[m], justify="right")
    def cell(v):
        return "–" if v is None else f"{v:.3f}" if isinstance(v, float | int) else str(v)

    for name, get in ANSWER_ROWS:
        t.add_row(name, *[cell(get(summary["modes"][m])) for m in modes])
    t.caption = "고친 코드로 다시 재려면 OPENAI_API_KEY를 넣고 서버를 켠 뒤 run_eval run → score (호출 한도 있음)"
    t.caption_justify = "left"
    return t


# ── 3. 한 번 실행: 재고 · 저장하고 · 표 출력 ─────────────────────────────

def once(split: str, modes: list[str]) -> None:
    last_path = OUT / f"live_last_{split}.json"   # 이전 실행 결과 (▲▼ 비교용, git에 올리지 않음)
    previous = json.loads(last_path.read_text(encoding="utf-8")) if last_path.exists() else None
    result = measure(split, modes)
    console.print(search_table(result, previous))
    answers = answer_table(split)
    if answers:
        console.print(answers)
    OUT.mkdir(parents=True, exist_ok=True)
    last_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


# ── 4. 감시 모드: 파일을 저장할 때마다 다시 재기 ─────────────────────────

WATCH = ["app/*.py", "eval/*.py", "data/evaluation/*.json"]
REBUILD = {"ingest.py", "index.py", "config.py"}   # 청크나 색인이 달라지는 파일 → 색인부터 다시 만든다


def snapshot() -> dict[Path, float]:
    return {p: p.stat().st_mtime for pattern in WATCH for p in ROOT.glob(pattern)}


def run_step(args: list[str]) -> bool:
    """매번 새 프로세스로 실행해 고친 코드를 확실히 다시 불러온다."""
    return subprocess.run([sys.executable, "-X", "utf8", *args], cwd=ROOT).returncode == 0


def watch(split: str, modes: list[str]) -> None:
    eval_args = ["-m", "eval.live", "--set", split, "--modes", ",".join(modes)]
    run_step(eval_args)
    seen = snapshot()
    console.print("[dim]파일 변경을 기다리는 중… (Ctrl+C로 끝)[/]")
    while True:
        time.sleep(1)
        now = snapshot()
        changed = [p for p in now if seen.get(p) != now[p]]
        if not changed:
            continue
        time.sleep(1.5)   # 저장이 여러 번 이어질 때 한 번만 돌도록 잠깐 기다린다
        now = snapshot()
        changed = sorted({p for p in now if seen.get(p) != now[p]} | set(changed))
        seen = now
        names = ", ".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in changed)
        console.rule(f"[bold]변경 감지 {datetime.now():%H:%M:%S}[/] · {names}")
        if any(p.name in REBUILD and p.parent.name == "app" for p in changed):
            console.print("청크 · 색인 코드가 바뀌어 색인을 다시 만듭니다 (python -m app.index)")
            if not run_step(["-m", "app.index"]):
                console.print("[red]색인 만들기 실패. 오류를 고치고 다시 저장하세요.[/]")
                continue
        if not run_step(eval_args):
            console.print("[red]평가 실패. 오류를 고치고 다시 저장하세요.[/]")
        console.print("[dim]파일 변경을 기다리는 중… (Ctrl+C로 끝)[/]")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="검색 평가지표를 터미널 표로 본다")
    parser.add_argument("--set", default="development", help="질문 묶음 (기본 development)")
    parser.add_argument("--modes", default=",".join(MODES), help="쉼표로 구분 (naive,hybrid,hybrid_rerank)")
    parser.add_argument("--watch", action="store_true", help="파일을 저장할 때마다 다시 잰다")
    a = parser.parse_args()
    chosen = [m for m in a.modes.split(",") if m in MODES]
    try:
        watch(a.set, chosen) if a.watch else once(a.set, chosen)
    except KeyboardInterrupt:
        pass
