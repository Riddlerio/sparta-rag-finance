# 세 RAG 방식 비교표 (개발 질문 30개)

k=10 · 입력 한도 3,000토큰 · 생성 gpt-6-luna · 채점 gpt-6-luna (RAGAS 0.4.3 Faithfulness + 맞춤 Judge)

## 방식별 평균

| 지표 | naive | hybrid | hybrid_rerank |
|---|---|---|---|
| Hit@10 | 0.929 (n=28) | 0.964 (n=28) | 0.929 (n=28) |
| Recall@10 | 0.833 | 0.878 | 0.860 |
| Faithfulness | 0.95 (n=28, 해당없음 1, 불가 1) | 0.95 (n=29, 해당없음 0, 불가 1) | 0.97 (n=29, 해당없음 0, 불가 1) |
| Judge 사실 | 0.67 (n=30, 해당없음 0, 불가 0) | 0.70 (n=30, 해당없음 0, 불가 0) | 0.87 (n=30, 해당없음 0, 불가 0) |
| Judge 조건 | 1 (n=30, 해당없음 0, 불가 0) | 0.93 (n=30, 해당없음 0, 불가 0) | 1 (n=30, 해당없음 0, 불가 0) |
| Judge 근거 | 0.87 (n=30, 해당없음 0, 불가 0) | 0.90 (n=30, 해당없음 0, 불가 0) | 1 (n=30, 해당없음 0, 불가 0) |
| Judge 판단 | 0.97 (n=30, 해당없음 0, 불가 0) | 0.90 (n=30, 해당없음 0, 불가 0) | 1 (n=30, 해당없음 0, 불가 0) |
| 평균 응답 시간 | 3,347ms (정상 30건) | 3,214ms (정상 30건) | 8,821ms (정상 30건) |
| 오류율 | 0.0% (0건) | 0.0% (0건) | 0.0% (0건) |
| 입력 / 출력 토큰 | 53,904 / 7,348 (30건 확인) | 53,907 / 7,423 (30건 확인) | 53,888 / 7,282 (30건 확인) |

채점 호출(서비스와 별도): Judge 90회, RAGAS 178회 · 비용: 확인 불가 (gpt-6-luna 단가를 확인하지 못함)

## 질문별 결과

| 질문 | 기대 | 방식 | 상태 | Hit | Recall | Faith | 사실 | 조건 | 근거 | 판단 | 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F-DEV-01 | answer | naive | answered | 1 | 0.50 | 1.00 | 1 | 1 | 1 | 1 | 3,228ms |
| F-DEV-01 | answer | hybrid | answered | 1 | 0.50 | 1.00 | 0 | 1 | 1 | 1 | 2,253ms |
| F-DEV-01 | answer | hybrid_rerank | answered | 1 | 0.50 | 1.00 | 1 | 1 | 1 | 1 | 11,644ms |
| F-DEV-02 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,654ms |
| F-DEV-02 | answer | hybrid | answered | 1 | 1.00 | 0.67 | 1 | 1 | 1 | 1 | 2,464ms |
| F-DEV-02 | answer | hybrid_rerank | answered | 1 | 1.00 | 0.80 | 1 | 1 | 1 | 1 | 7,891ms |
| F-DEV-03 | answer | naive | answered | 0 | 0.00 | 1.00 | 0 | 1 | 1 | 1 | 2,446ms |
| F-DEV-03 | answer | hybrid | answered | 1 | 1.00 | 0.80 | 0 | 1 | 0 | 1 | 3,277ms |
| F-DEV-03 | answer | hybrid_rerank | answered | 0 | 0.00 | 1.00 | 1 | 1 | 1 | 1 | 7,446ms |
| F-DEV-04 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,119ms |
| F-DEV-04 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,751ms |
| F-DEV-04 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 6,918ms |
| F-DEV-05 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,257ms |
| F-DEV-05 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,057ms |
| F-DEV-05 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 9,119ms |
| F-DEV-06 | answer | naive | abstained | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 4,351ms |
| F-DEV-06 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,292ms |
| F-DEV-06 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 7,364ms |
| F-DEV-07 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,717ms |
| F-DEV-07 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,637ms |
| F-DEV-07 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 7,840ms |
| F-DEV-08 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 1,986ms |
| F-DEV-08 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,287ms |
| F-DEV-08 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 6,896ms |
| F-DEV-09 | answer | naive | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 3,916ms |
| F-DEV-09 | answer | hybrid | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 4,420ms |
| F-DEV-09 | answer | hybrid_rerank | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 9,889ms |
| F-DEV-10 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,027ms |
| F-DEV-10 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 1,950ms |
| F-DEV-10 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 7,943ms |
| F-DEV-11 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,274ms |
| F-DEV-11 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,373ms |
| F-DEV-11 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 7,953ms |
| F-DEV-12 | abstain | naive | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 2,697ms |
| F-DEV-12 | abstain | hybrid | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 2,388ms |
| F-DEV-12 | abstain | hybrid_rerank | abstained | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 8,089ms |
| F-DEV-13 | answer | naive | answered | 1 | 0.33 | 1.00 | 1 | 1 | 1 | 1 | 2,927ms |
| F-DEV-13 | answer | hybrid | abstained | 1 | 0.33 | 1.00 | 0 | 1 | 1 | 1 | 4,060ms |
| F-DEV-13 | answer | hybrid_rerank | answered | 1 | 0.33 | 1.00 | 1 | 1 | 1 | 1 | 8,820ms |
| F-DEV-14 | answer | naive | answered | 1 | 1.00 | 0.67 | 1 | 1 | 1 | 1 | 5,728ms |
| F-DEV-14 | answer | hybrid | answered | 1 | 1.00 | – | 1 | 1 | 1 | 1 | 5,237ms |
| F-DEV-14 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 10,419ms |
| F-DEV-15 | answer | naive | answered | 1 | 1.00 | – | 0 | 1 | 0 | 1 | 4,074ms |
| F-DEV-15 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,829ms |
| F-DEV-15 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 9,385ms |
| F-DEV-16 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 3,927ms |
| F-DEV-16 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 3,300ms |
| F-DEV-16 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 8,908ms |
| F-DEV-17 | answer | naive | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 3,111ms |
| F-DEV-17 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 3,077ms |
| F-DEV-17 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 9,548ms |
| F-DEV-18 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,404ms |
| F-DEV-18 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,692ms |
| F-DEV-18 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 8,956ms |
| F-DEV-19 | clarify | naive | needs_clarification | – | – | – | 0 | 1 | 0 | 0 | 2,892ms |
| F-DEV-19 | clarify | hybrid | answered | – | – | 1.00 | 1 | 0 | 1 | 0 | 3,931ms |
| F-DEV-19 | clarify | hybrid_rerank | needs_clarification | – | – | 1.00 | 1 | 1 | 1 | 1 | 7,665ms |
| F-DEV-20 | abstain | naive | abstained | 0 | 0.00 | 1.00 | 1 | 1 | 1 | 1 | 3,146ms |
| F-DEV-20 | abstain | hybrid | abstained | 0 | 0.00 | 1.00 | 1 | 1 | 1 | 1 | 2,364ms |
| F-DEV-20 | abstain | hybrid_rerank | abstained | 0 | 0.00 | 1.00 | 1 | 1 | 1 | 1 | 7,634ms |
| F-DEV-21 | answer | naive | abstained | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 6,487ms |
| F-DEV-21 | answer | hybrid | abstained | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 0 | 5,881ms |
| F-DEV-21 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 7,930ms |
| F-DEV-22 | answer | naive | abstained | 1 | 0.50 | 1.00 | 0 | 1 | 1 | 1 | 3,092ms |
| F-DEV-22 | answer | hybrid | abstained | 1 | 0.50 | 1.00 | 0 | 1 | 0 | 1 | 2,859ms |
| F-DEV-22 | answer | hybrid_rerank | abstained | 1 | 0.50 | 1.00 | 0 | 1 | 1 | 1 | 7,920ms |
| F-DEV-23 | answer | naive | answered | 1 | 1.00 | 0.86 | 1 | 1 | 1 | 1 | 4,432ms |
| F-DEV-23 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 5,372ms |
| F-DEV-23 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 8,900ms |
| F-DEV-24 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,863ms |
| F-DEV-24 | answer | hybrid | answered | 1 | 1.00 | 0.80 | 1 | 1 | 1 | 1 | 3,502ms |
| F-DEV-24 | answer | hybrid_rerank | abstained | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 12,526ms |
| F-DEV-25 | answer | naive | abstained | 1 | 0.50 | 0.88 | 0 | 1 | 0 | 1 | 5,360ms |
| F-DEV-25 | answer | hybrid | abstained | 1 | 0.50 | 0.67 | 0 | 1 | 1 | 1 | 4,879ms |
| F-DEV-25 | answer | hybrid_rerank | abstained | 1 | 1.00 | – | 0 | 1 | 1 | 1 | 11,982ms |
| F-DEV-26 | answer | naive | abstained | 1 | 0.50 | 0.83 | 0 | 1 | 0 | 1 | 4,318ms |
| F-DEV-26 | answer | hybrid | answered | 1 | 0.75 | 0.80 | 0 | 0 | 0 | 0 | 4,279ms |
| F-DEV-26 | answer | hybrid_rerank | abstained | 1 | 0.75 | 1.00 | 0 | 1 | 1 | 1 | 8,834ms |
| F-DEV-27 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 4,235ms |
| F-DEV-27 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 2,212ms |
| F-DEV-27 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 7,756ms |
| F-DEV-28 | answer | naive | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 3,331ms |
| F-DEV-28 | answer | hybrid | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 3,775ms |
| F-DEV-28 | answer | hybrid_rerank | answered | 1 | 1.00 | 1.00 | 1 | 1 | 1 | 1 | 9,287ms |
| F-DEV-29 | answer | naive | answered | 1 | 1.00 | 1.00 | 0 | 1 | 1 | 1 | 2,661ms |
| F-DEV-29 | answer | hybrid | answered | 1 | 1.00 | 0.80 | 1 | 1 | 1 | 1 | 2,342ms |
| F-DEV-29 | answer | hybrid_rerank | answered | 1 | 1.00 | 0.25 | 1 | 1 | 1 | 1 | 9,591ms |
| F-DEV-30 | clarify | naive | needs_clarification | – | – | 0.50 | 1 | 1 | 1 | 1 | 2,754ms |
| F-DEV-30 | clarify | hybrid | needs_clarification | – | – | 1.00 | 1 | 1 | 1 | 1 | 2,674ms |
| F-DEV-30 | clarify | hybrid_rerank | needs_clarification | – | – | 1.00 | 1 | 1 | 1 | 1 | 9,583ms |
