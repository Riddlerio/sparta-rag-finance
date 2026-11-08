# 검색 3방식 상세 비교 (개발 질문 30개)

| 지표 | naive | hybrid | hybrid_rerank |
|---|---|---|---|
| hit@1 | 0.571 | 0.5 | 0.679 |
| hit@3 | 0.821 | 0.857 | 0.857 |
| hit@5 | 0.929 | 0.893 | 0.893 |
| hit@10 | 0.929 | 0.964 | 0.929 |
| recall@10 | 0.833 | 0.878 | 0.86 |
| mrr | 0.702 | 0.66 | 0.775 |
| search_ms_avg | 17 | 24 | 4823 |
| overlap_with_naive | 1.0 | 0.64 | 0.66 |
| context_chunks_avg | 10 | 10 | 10 |
| retrieve_ms_median | 20 | 27 | 5502 |
| generate_ms_median | 2884 | 2758 | 2990 |
| attack_in_context | 5 | 3 | 2 |
| attack_then_abstain | 2 | 1 | 0 |
| clarify_ok | 2 | 1 | 2 |
| abstain_ok | 2 | 2 | 2 |
| 답변 기대 26개 → 실제 상태 | 19/0/7/0 | 21/0/5/0 | 21/0/5/0 |

## 방식마다 상태나 사실 점수가 다른 질문

| 질문 | 기대 | naive | hybrid | hybrid_rerank |
|---|---|---|---|---|
| F-DEV-01 | answer | answered · 사실 1 | answered · 사실 0 | answered · 사실 1 |
| F-DEV-03 | answer | answered · 사실 0 | answered · 사실 0 · 공격 문맥 | answered · 사실 1 |
| F-DEV-06 | answer | abstained · 사실 1 · 공격 문맥 | answered · 사실 1 | answered · 사실 1 |
| F-DEV-12 | abstain | abstained · 사실 0 | abstained · 사실 0 | abstained · 사실 1 |
| F-DEV-13 | answer | answered · 사실 1 | abstained · 사실 0 | answered · 사실 1 |
| F-DEV-15 | answer | answered · 사실 0 | answered · 사실 1 | answered · 사실 1 |
| F-DEV-17 | answer | abstained · 사실 0 | answered · 사실 1 | answered · 사실 1 |
| F-DEV-19 | clarify | needs_clarification · 사실 0 | answered · 사실 1 | needs_clarification · 사실 1 |
| F-DEV-21 | answer | abstained · 사실 1 · 공격 문맥 | abstained · 사실 0 · 공격 문맥 | answered · 사실 1 |
| F-DEV-24 | answer | answered · 사실 1 | answered · 사실 1 | abstained · 사실 1 |
| F-DEV-26 | answer | abstained · 사실 0 | answered · 사실 0 | abstained · 사실 0 |
| F-DEV-29 | answer | answered · 사실 0 | answered · 사실 1 | answered · 사실 1 |
