# 정답 청크(quote) 기준 채점 (development)

정답 청크 = 정답 파일 quote가 든 청크 (공백 · 문장부호를 지우고 비교). 모델 호출 없이 저장된 결과만 다시 읽었다.

| 지표 | naive | hybrid | hybrid_rerank |
|---|---|---|---|
| 문맥에 정답 청크 (Hit) | 0.607 | 0.607 | 0.714 |
| 문맥의 근거 묶음 Recall | 0.411 | 0.449 | 0.47 |
| 검색 순위 chunk_hit@1 | 0.179 | 0.179 | 0.5 |
| 검색 순위 chunk_hit@3 | 0.393 | 0.464 | 0.607 |
| 검색 순위 chunk_mrr | 0.309 | 0.318 | 0.571 |
| 검색 순위 group_recall@10 | 0.411 | 0.449 | 0.47 |

## 실패 분류 (정답 청크 확보 × Judge 사실)

| 분류 | naive | hybrid | hybrid_rerank |
|---|---|---|---|
| 검색 · 문맥 실패 | 3 | 5 | 1 |
| 근거 밖에서 맞힘 | 8 | 6 | 7 |
| 근거 일부 누락 | 5 | 3 | 3 |
| 생성 실패 | 1 | 1 | 0 |
| 성공 | 6 | 8 | 7 |
| 일부 근거로 맞힘 | 5 | 5 | 10 |
| 해당 없음 | 2 | 2 | 2 |

## 성공이 아닌 질문

| 질문 | 방식 | 상태 | 사실 | 묶음 Recall | 분류 | 빠진 정답 청크 |
|---|---|---|---|---|---|---|
| F-DEV-01 | naive | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB02-p1-c2 |
| F-DEV-03 | naive | answered | 0 | 0.00 | 검색 · 문맥 실패 | FB02-p1-c1, FB02-p1-c3, FB02-p1-c2 |
| F-DEV-07 | naive | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB03-p1-c2 |
| F-DEV-08 | naive | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB10-p2-c3 |
| F-DEV-09 | naive | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB10-p2-c1, FB10-p2-c2, FB10-p2-c3 |
| F-DEV-12 | naive | abstained | 0 | 1.00 | 생성 실패 |  |
| F-DEV-13 | naive | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB01-p2-c2, FB04-p2-c2, FB05-p1-c2 |
| F-DEV-14 | naive | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB07-p1-c1, FB07-p1-c2, FB08-p1-c1, FB08-p1-c2 |
| F-DEV-15 | naive | answered | 0 | 0.33 | 근거 일부 누락 | FB10-p2-c1, FB10-p2-c3 |
| F-DEV-16 | naive | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB07-p1-c1, FB07-p1-c2 |
| F-DEV-17 | naive | abstained | 0 | 0.33 | 근거 일부 누락 | FB10-p2-c1, FB07-p2-c1 |
| F-DEV-18 | naive | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB10-p2-c2 |
| F-DEV-20 | naive | abstained | 1 | 0.00 | 근거 밖에서 맞힘 | FB02-p4-c4 |
| F-DEV-21 | naive | abstained | 1 | 0.00 | 근거 밖에서 맞힘 | FB04-p3-c1, FB05-p1-c2 |
| F-DEV-22 | naive | abstained | 0 | 0.50 | 근거 일부 누락 | FB04-p2-c1 |
| F-DEV-23 | naive | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB03-p1-c2, FB03-p1-c3 |
| F-DEV-24 | naive | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB10-p2-c1, FB04-p3-c1 |
| F-DEV-25 | naive | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB03-p1-c1, FB03-p1-c4, FB06-p1-c1, FB06-p1-c4 |
| F-DEV-26 | naive | abstained | 0 | 0.33 | 근거 일부 누락 | FB02-p2-c1/FB09-p1-c2, FB06-p4-c1/FB09-p2-c3 |
| F-DEV-27 | naive | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB05-p1-c2 |
| F-DEV-28 | naive | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB10-p2-c1, FB10-p2-c3 |
| F-DEV-29 | naive | answered | 0 | 0.50 | 근거 일부 누락 | FB03-p3-c3 |
| F-DEV-01 | hybrid | answered | 0 | 0.50 | 근거 일부 누락 | FB02-p1-c2 |
| F-DEV-03 | hybrid | answered | 0 | 0.00 | 검색 · 문맥 실패 | FB02-p1-c1, FB02-p1-c3, FB02-p1-c2 |
| F-DEV-07 | hybrid | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB03-p1-c2 |
| F-DEV-08 | hybrid | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB10-p2-c3 |
| F-DEV-09 | hybrid | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB10-p2-c1, FB10-p2-c2, FB10-p2-c3 |
| F-DEV-12 | hybrid | abstained | 0 | 1.00 | 생성 실패 |  |
| F-DEV-13 | hybrid | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB01-p2-c2, FB04-p2-c2, FB05-p1-c2 |
| F-DEV-14 | hybrid | answered | 1 | 0.25 | 일부 근거로 맞힘 | FB07-p1-c1, FB07-p1-c2, FB08-p1-c1 |
| F-DEV-15 | hybrid | answered | 1 | 0.33 | 일부 근거로 맞힘 | FB10-p2-c1, FB10-p2-c3 |
| F-DEV-16 | hybrid | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB07-p1-c1, FB07-p1-c2 |
| F-DEV-17 | hybrid | answered | 1 | 0.33 | 일부 근거로 맞힘 | FB10-p2-c1, FB10-p2-c4 |
| F-DEV-18 | hybrid | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB10-p2-c2, FB10-p2-c3 |
| F-DEV-20 | hybrid | abstained | 1 | 0.00 | 근거 밖에서 맞힘 | FB02-p4-c4 |
| F-DEV-21 | hybrid | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB04-p3-c1, FB05-p1-c2 |
| F-DEV-22 | hybrid | abstained | 0 | 0.50 | 근거 일부 누락 | FB04-p2-c1 |
| F-DEV-23 | hybrid | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB03-p1-c2, FB03-p1-c3 |
| F-DEV-24 | hybrid | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB10-p2-c1, FB04-p3-c1 |
| F-DEV-25 | hybrid | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB03-p1-c1, FB03-p1-c4, FB06-p1-c1, FB06-p1-c4 |
| F-DEV-26 | hybrid | answered | 0 | 0.67 | 근거 일부 누락 | FB02-p2-c1/FB09-p1-c2 |
| F-DEV-28 | hybrid | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB10-p2-c1, FB10-p2-c3 |
| F-DEV-01 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB02-p1-c2 |
| F-DEV-03 | hybrid_rerank | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB02-p1-c1, FB02-p1-c3, FB02-p1-c2 |
| F-DEV-06 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB05-p1-c3 |
| F-DEV-07 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB03-p1-c2 |
| F-DEV-08 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB10-p2-c3 |
| F-DEV-09 | hybrid_rerank | abstained | 0 | 0.00 | 검색 · 문맥 실패 | FB10-p2-c1, FB10-p2-c2, FB10-p2-c3 |
| F-DEV-13 | hybrid_rerank | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB01-p2-c2, FB04-p2-c2, FB05-p1-c2 |
| F-DEV-14 | hybrid_rerank | answered | 1 | 0.25 | 일부 근거로 맞힘 | FB07-p1-c1, FB07-p1-c2, FB08-p1-c1 |
| F-DEV-15 | hybrid_rerank | answered | 1 | 0.33 | 일부 근거로 맞힘 | FB10-p2-c1, FB10-p2-c3 |
| F-DEV-16 | hybrid_rerank | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB07-p1-c1, FB07-p1-c2 |
| F-DEV-17 | hybrid_rerank | answered | 1 | 0.67 | 일부 근거로 맞힘 | FB10-p2-c1 |
| F-DEV-18 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB10-p2-c2 |
| F-DEV-20 | hybrid_rerank | abstained | 1 | 0.00 | 근거 밖에서 맞힘 | FB02-p4-c4 |
| F-DEV-21 | hybrid_rerank | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB04-p3-c1, FB05-p1-c2 |
| F-DEV-22 | hybrid_rerank | abstained | 0 | 0.50 | 근거 일부 누락 | FB04-p2-c1 |
| F-DEV-23 | hybrid_rerank | answered | 1 | 0.00 | 근거 밖에서 맞힘 | FB03-p1-c2, FB03-p1-c3 |
| F-DEV-24 | hybrid_rerank | abstained | 1 | 0.00 | 근거 밖에서 맞힘 | FB10-p2-c1, FB04-p3-c1 |
| F-DEV-25 | hybrid_rerank | abstained | 0 | 0.25 | 근거 일부 누락 | FB03-p1-c1, FB03-p1-c4, FB06-p1-c1 |
| F-DEV-26 | hybrid_rerank | abstained | 0 | 0.67 | 근거 일부 누락 | FB02-p2-c1/FB09-p1-c2 |
| F-DEV-27 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB05-p1-c2 |
| F-DEV-28 | hybrid_rerank | answered | 1 | 0.50 | 일부 근거로 맞힘 | FB10-p2-c3 |
