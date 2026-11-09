"""검색 3방식(naive · hybrid · hybrid_rerank)이 어디서 갈리는지 자세히 비교한다.

1) 검색만 다시 돌려 순위를 잰다 (모델 호출 없음)
   - Hit@1 · 3 · 5 · 10: 상위 k개 안에 정답 근거 문서가 있나
   - Recall@10: 정답 근거 문서 중 몇 개를 찾았나
   - MRR: 첫 정답 문서가 몇 등에 나왔나 (1등 = 1, 2등 = 0.5 …)
   - 청크 단위 Hit@1 · 3 · 10 · MRR · 근거 묶음 Recall: 정답 파일 quote가 든 청크 기준
   - 검색 시간, BM25만 찾은 청크 수, 리랭커가 바꾼 청크 수
2) 이미 저장한 답변 90개(run_eval 결과)를 방식별로 다시 읽는다 (추가 호출 없음)

정답 파일은 검색이 끝난 뒤 채점할 때만 연다. 공격 문장은 "문맥에 들어갔는지" 세는 데만 쓴다.

실행 (서버를 끄고: Qdrant 폴더는 한 프로그램만 열 수 있다)
- python -m eval.compare_modes               → eval/results/mode_comparison.md · .json
- python -m eval.compare_modes show "질문"   → 질문 하나를 세 방식으로 검색해 상위 5개씩 출력
"""
import json
import sys
import time
from statistics import mean, median

from app.config import CANDIDATES, CHUNKS_PATH, EVAL_DIR, ROOT, TOP_K
from app.index import load_resources
from app.retrieve import PIPELINES, bm25_search, dense_search
from eval.run_eval import chunk_rank_metrics, gold_books
from eval.security_eval import attack_chunks

MODES = list(PIPELINES)
OUT = ROOT / "eval" / "results"
EXPECTED = {"answer": "answered", "clarify": "needs_clarification", "abstain": "abstained"}


def show(question: str) -> None:
    """질문 하나를 세 방식으로 검색해 상위 5개를 나란히 보여 준다."""
    res = load_resources()
    for mode, search in PIPELINES.items():
        t0 = time.perf_counter()
        ranked = search(res, question)
        print(f"\n[{mode}] {(time.perf_counter() - t0) * 1000:.0f}ms")
        for i, (cid, score) in enumerate(ranked[:5], start=1):
            print(f"  {i}. {cid:<14} {score:8.4f}  {res.by_id[cid]['text'][:60]}")


def search_all(res, questions: list[dict]) -> dict:
    """질문마다 세 방식으로 검색하고, 결과 · 시간 · 차이가 난 청크를 모은다."""
    PIPELINES["hybrid_rerank"](res, "준비")  # 리랭커를 미리 불러 첫 질문 시간이 부풀지 않게 한다
    rows = {}
    for q in questions:
        row = {"ranked": {}, "ms": {}}
        for mode, search in PIPELINES.items():
            t0 = time.perf_counter()
            row["ranked"][mode] = [cid for cid, _ in search(res, q["question"])]
            row["ms"][mode] = (time.perf_counter() - t0) * 1000
        dense20 = {cid for cid, _ in dense_search(res, q["question"], CANDIDATES)}
        bm25_20 = {cid for cid, _ in bm25_search(res, q["question"], CANDIDATES)}
        row["bm25_only_in_hybrid"] = [c for c in row["ranked"]["hybrid"] if c in bm25_20 and c not in dense20]
        row["rerank_new"] = [c for c in row["ranked"]["hybrid_rerank"] if c not in row["ranked"]["hybrid"]]
        rows[q["id"]] = row
    return rows


def first_gold_rank(books: list[str], gold: set[str]) -> int | None:
    """정답 근거 문서가 처음 나온 순위 (없으면 None)."""
    return next((i for i, b in enumerate(books, start=1) if b in gold), None)


def rank_metrics(res, rows: dict, answers: dict) -> dict:
    """방식별 순위 지표. 정답 근거 문서가 없는 질문(보류가 정답인 질문)은 빼고 계산한다."""
    scored = [qid for qid in rows if gold_books(answers[qid])]
    by_page = {}
    for c in res.chunks:
        by_page.setdefault((c["book_id"], c["page"]), []).append(c)
    out = {}
    for mode in MODES:
        ranks, recalls = [], []
        for qid in scored:
            gold = gold_books(answers[qid])
            books = [res.by_id[c]["book_id"] for c in rows[qid]["ranked"][mode]]
            ranks.append(first_gold_rank(books, gold))
            recalls.append(len(gold & set(books)) / len(gold))
        ms = [rows[qid]["ms"][mode] for qid in rows]
        overlap = [len(set(r["ranked"][mode]) & set(r["ranked"]["naive"])) / TOP_K for r in rows.values()]
        out[mode] = {
            **{f"hit@{k}": round(mean(r is not None and r <= k for r in ranks), 3) for k in (1, 3, 5, TOP_K)},
            f"recall@{TOP_K}": round(mean(recalls), 3),
            "mrr": round(mean(1 / r if r else 0 for r in ranks), 3),
            "n": len(scored), "na": len(rows) - len(scored),
            "search_ms_avg": round(mean(ms)), "search_ms_max": round(max(ms)),
            "overlap_with_naive": round(mean(overlap), 2),
            "first_gold_rank": dict(zip(scored, ranks)),
            **chunk_rank_metrics({qid: rows[qid]["ranked"][mode] for qid in rows}, answers, by_page),
        }
    out["hybrid"]["bm25_only_chunks_avg"] = round(mean(len(r["bm25_only_in_hybrid"]) for r in rows.values()), 1)
    out["hybrid_rerank"]["new_vs_hybrid_avg"] = round(mean(len(r["rerank_new"]) for r in rows.values()), 1)
    return out


def answer_metrics(answers: dict) -> dict:
    """저장된 답변 90개를 방식별로 다시 센다: 기대 행동과 실제 상태, 공격 문단 노출, 단계별 시간."""
    runs = [json.loads(line) for line in (OUT / "runs_development.jsonl").open(encoding="utf-8")]
    scores = {(s["id"], s["mode"]): s for s in map(json.loads, (OUT / "scores_development.jsonl").open(encoding="utf-8"))}
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    tasks = json.loads((EVAL_DIR / "security_tasks.json").read_text(encoding="utf-8"))
    attack_ids = {cid for t in tasks for cid in attack_chunks(chunks, t["attack_text"])}

    by_mode, per_question = {}, {}
    for mode in MODES:
        mine = [r for r in runs if r["mode"] == mode]
        step_ms = lambda r, name: next(s["ms"] for s in r["steps"] if s["step"] == name)
        m = {"answer_expected": {"answered": 0, "needs_clarification": 0, "abstained": 0, "error": 0},
             "clarify_ok": 0, "abstain_ok": 0, "attack_in_context": 0, "attack_then_abstain": 0,
             "context_chunks_avg": round(mean(len(r["contexts"]) for r in mine), 1),
             "retrieve_ms_median": round(median(step_ms(r, "retrieve") for r in mine)),
             "generate_ms_median": round(median(step_ms(r, "generate") for r in mine))}
        for r in mine:
            expected = answers[r["id"]]["expected_action"]
            if expected == "answer":
                m["answer_expected"][r["status"]] += 1
            elif r["status"] == EXPECTED[expected]:
                m[f"{expected}_ok"] += 1
            attacked = any(c["chunk_id"] in attack_ids for c in r["contexts"])
            m["attack_in_context"] += attacked
            m["attack_then_abstain"] += attacked and r["status"] == "abstained"
            judge = scores[(r["id"], mode)]["judge"]
            per_question.setdefault(r["id"], {"expected": expected})[mode] = {
                "status": r["status"], "fact": judge["fact"]["score"], "evidence": judge["evidence"]["score"],
                "judgment": judge["judgment"]["score"], "attack": attacked, "contexts": len(r["contexts"])}
        by_mode[mode] = m
    # 방식마다 상태나 사실 점수가 다른 질문만 따로 모은다
    differs = {qid: v for qid, v in per_question.items() if len({(v[m]["status"], v[m]["fact"]) for m in MODES}) > 1}
    return {"by_mode": by_mode, "differs": differs}


def write_md(rank: dict, ans: dict) -> None:
    """사람이 읽을 비교표를 쓴다."""
    def row(name: str, values: list) -> str:
        return f"| {name} | " + " | ".join(str(v) for v in values) + " |"

    lines = ["# 검색 3방식 상세 비교 (개발 질문 30개)", "", row("지표", MODES), "|---|" + "---|" * len(MODES)]
    for key in ("hit@1", "hit@3", "hit@5", f"hit@{TOP_K}", f"recall@{TOP_K}", "mrr",
                "chunk_hit@1", "chunk_hit@3", "chunk_hit@10", "chunk_mrr", "group_recall@10", "search_ms_avg", "overlap_with_naive"):
        lines.append(row(key, [rank[m][key] for m in MODES]))
    for key in ("context_chunks_avg", "retrieve_ms_median", "generate_ms_median",
                "attack_in_context", "attack_then_abstain", "clarify_ok", "abstain_ok"):
        lines.append(row(key, [ans["by_mode"][m][key] for m in MODES]))
    lines.append(row("답변 기대 26개 → 실제 상태",
                     ["/".join(map(str, ans["by_mode"][m]["answer_expected"].values())) for m in MODES]))

    lines += ["", "## 방식마다 상태나 사실 점수가 다른 질문", "",
              "| 질문 | 기대 | " + " | ".join(MODES) + " |", "|---|---|" + "---|" * len(MODES)]
    for qid, v in sorted(ans["differs"].items()):
        cells = [f"{v[m]['status']} · 사실 {v[m]['fact']}" + (" · 공격 문맥" if v[m]["attack"] else "") for m in MODES]
        lines.append(f"| {qid} | {v['expected']} | " + " | ".join(cells) + " |")
    (OUT / "mode_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    res = load_resources()
    questions = json.loads((EVAL_DIR / "development_questions.json").read_text(encoding="utf-8"))
    rows = search_all(res, questions)  # 검색에는 질문만 쓴다
    answers = {a["id"]: a for a in json.loads((EVAL_DIR / "development.json").read_text(encoding="utf-8"))}
    rank, ans = rank_metrics(res, rows, answers), answer_metrics(answers)
    OUT.mkdir(parents=True, exist_ok=True)
    per_question = {qid: {k: v for k, v in r.items() if k != "ms"} for qid, r in rows.items()}
    (OUT / "mode_comparison.json").write_text(json.dumps(
        {"rank": rank, "answers": ans, "per_question_search": per_question}, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(rank, ans)
    print((OUT / "mode_comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 2 and sys.argv[1] == "show":
        show(sys.argv[2])
    else:
        main()
