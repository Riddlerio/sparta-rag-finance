# 프로젝트 규칙

진행하면서 합의한 규칙을 여기에 쌓아 갑니다. 규칙마다 근거(발제문 / 팀 합의)를 함께 적습니다.

## 공통

1. **없는 것은 지어내지 않는다.** 자료에 없는 정보는 보충하거나 추측하지 않고, "확인할 수 없다/없다"고 사실대로 내보낸다. (팀 합의)

## 청킹

### 발제문이 정해 둔 조건

- `documents/*.pdf`만 색인한다. 평가 JSON(질문·정답·채점 기준)은 색인하지 않는다.
- `book_id` = 확장자를 뺀 파일명 (`FB01.pdf` → `FB01`). `file_id`를 쓰면 같은 값.
- 페이지는 PDF 뷰어 기준, 첫 페이지 = 1쪽.
- 원문에 없는 메타데이터는 임의로 만들지 않는다.
- 청크를 나눠도 대상, 적용일, 단위, 출처가 서로 떨어지지 않게 보존한다.
- 여러 페이지를 합친 청크는 한 페이지처럼 표시하지 않고 각 구절의 페이지를 남긴다.
- 표의 값은 행·열 제목, 단위, 각주와 함께 보존한다.
- 세 RAG 방식은 같은 PDF, 같은 청크, 같은 임베딩을 쓴다. (개발 중엔 조정 가능, 최종 비교 땐 고정)
- 청크 크기·overlap 조정은 대표 자료에서 하나씩 바꿔 근거가 남는지 확인하는 방식. 별도 필수 실험은 아님.

발제문 청크 예시 (형식 참고용, 가상 자료):

```json
{
  "page_content": "2026년 11월 1일부터 장비 A의 자동 화면 꺼짐 시간을 15분으로 정정합니다. ...",
  "metadata": {
    "book_id": "DEMO_NOTICE",
    "file_name": "DEMO_NOTICE.pdf",
    "page": 1,
    "chunk_id": "DEMO_NOTICE-p1-c1"
  }
}
```

### 자료 관찰 (2026-11-07 측정)

- 모든 페이지 구조: 머리말(`FB01 | 부서`) → 문서 제목 → 섹션 제목 → 문단 5~6개 → 꼬리말(`n / 24`)
- 360쪽 모두 완결된 문장으로 끝남 (페이지를 넘어가는 문장 없음). 문서 안에서 섹션 제목 중복 없음.
- 문단 경계는 pypdf `extraction_mode="layout"`의 빈 줄로 구분 가능.
- 문단 1,992개: 109~225자, E5 토큰 53~119 (중앙값 89). E5 small/base/large 토크나이저 결과 동일.
- 문서 제목 + 섹션 제목: E5 토큰 14~28.
- 페이지 통째: E5 토큰 364~639, 360쪽 중 192쪽이 E5 입력 한도 512를 넘음.
- 앞 문단을 가리키며 시작하는 문단("이번", "반대로" 등): 15개 (0.8%).
- 줄바꿈에서 단어가 잘림(`매출`↵`은`). 원래 띄어쓰기 자리에서의 줄바꿈(`전년`↵`동기`)과 텍스트로는 구분 불가.
- 생성 모델 gpt-6-luna의 토크나이저와 최대 컨텍스트 길이는 아직 확인하지 못함.

### 발제문에 없는 것 → 팀이 정할 것

- [x] 청크를 나누는 단위 → **문단 단위를 1차 기준으로 한다.** (팀 합의, 2026-11-07)
- [ ] 최종 청크 길이: 전체 컨텍스트 길이와 overlap 길이를 함께 고려해 정한다. (팀 합의된 방향, 값은 미정)
- [x] 줄 잇기 규칙 → **빈 문자열로 이어붙임 유지** (팀 합의, 2026-11-08). `전년`↵`동기`가 붙는 한계는 알고 둔다.
- [ ] overlap — **지금은 없음으로 두고 비교안(앞 문단 마지막 1문장)은 나중에** (팀 합의, 2026-11-08). 근거 정리 완료(2026-11-07, 가이드맵 "overlap 근거"). 후보: A 없음 / B 앞 문단 마지막 문장 / C 1문단 겹치기 / D 검색 후 앞뒤 문단 붙이기. 문단(약 90토큰) 단위로 overlap 유무를 직접 비교한 실험은 찾지 못함 → 개발 질문으로 확인
- [ ] chunk_id 형식 (발제문 예시 `FB01-p2-c1` vs 강의 `문서ID:본문해시12:size:overlap:start`)
- [ ] BM25 토큰화 방식 — 정규식 토큰이면 "매출"로 찾히는 청크 300개 vs "매출" 글자가 든 청크 651개 (조사 붙은 형태가 따로 저장됨)
- [ ] 페이지 경계를 넘는 청크를 허용할지
- [ ] 한 청크에 반드시 함께 있어야 하는 정보 (금융: 회사, 보고기간, 사업부문·제품, 수치의 단위, 정정 여부)
- [ ] 메타데이터에 넣을 항목 (필수 외에 무엇을, 원문에서 확인 가능한 것만)

## 서비스 구조 (FastAPI + LangChain)

### 발제문이 정해 둔 조건

- FastAPI `/query`에서 질문을 받고 빈 질문과 지원하지 않는 검색 방식을 확인한다. `/health`는 서비스 응답 확인용.
- 요청 ID는 서버에서 만들어 실행 기록과 연결한다.
- 검색 문맥과 질문을 `ChatPromptTemplate`에 넣고, LCEL 또는 Runnable로 모델 호출과 출력 검증을 연결할 수 있다.
- 세 검색 방식은 같은 API에서 바꿔 실행한다(서비스 3개를 따로 만들지 않음). k와 입력 토큰 한도는 세 방식에서 같다.
- Langfuse 연동은 필수. 대화 메모리와 별도 사용자 화면은 선택.

### 팀이 정한 것

- [x] **체인 구성 → 안 1: 상태 dict를 넘기는 4단계 체인.** (팀 합의, 2026-11-08)
  - 체인 밖(FastAPI): `QueryRequest` 검사(실패 시 422, 체인 실행 안 함), request_id 생성, HTTP 상태, `QueryResponse`.
  - 서버 시작 때 한 번(lifespan): 임베딩 모델, 색인, 리랭커, 체인 객체.
  - 체인 안: `retrieve`(mode별 검색) → `pack_context`(중복 제거·S1..Sk 라벨·토큰 예산) → `generate`(`ChatPromptTemplate` + 구조화 출력 `LLMAnswer`) → `finish`(S-ID 검사·출처 변환·status). 각 단계는 우리 함수를 `RunnableLambda`로 감싼다.
  - 근거
    1. 발제가 요구한 LangChain 범위(프롬프트에 문맥·질문 넣기 → 모델 호출 → 출력 검증)와 정확히 맞는다.
    2. 강의 6-2 체인(`prepare | invoke_model | finish`)과 같은 모양이라 마감(11/13)까지 F-DEV-01 하나를 바로 끝까지 이을 수 있다.
    3. 검색된 chunk_id와 최종 문맥·모델 입출력이 상태 dict에 남아 Hit@k·Recall@k 계산과 Langfuse 기록에 그대로 쓴다.
    4. 길이 예산·출처 변환·error 판정이 우리 함수라 발표에서 결정 하나를 코드 한 곳으로 설명할 수 있다.
  - 대안(안 2, LangChain retriever + `configurable_alternatives`)은 나중에 `retrieve`만 바꿔 넘어갈 수 있다.
- 길이 규칙: `gpt-6-luna 컨텍스트 한도 ≥ 지시문 + 질문 (+ 이전 대화) + 문맥 + 답변 몫(max_tokens)`. 문맥은 남은 예산만큼 청크를 통째로 넣고, 안 들어가는 청크는 자르지 않고 건너뛴다(강의 6-1).

### 팀이 정할 것

- [ ] Pydantic 모델 3개(`QueryRequest`, `LLMAnswer`, `QueryResponse`)의 필드
- [ ] 상태 dict의 키 (Langfuse에 남길 항목: 질문, 검색 문서, 최종 문맥, 모델 입출력, 시간, 토큰)
- [x] 입력 토큰 한도 **3,000** (팀 합의, 2026-11-08). k=10, max_tokens 800. 실측 입력 약 1,800토큰이라 10개 청크가 모두 들어가고, 세 방식에 같은 조건이 유지된다.
- [x] error일 때 HTTP 상태 코드 → **B안: 5xx + 같은 본문** (팀 합의, 2026-11-08). 모델 호출 실패·출력 검증 실패 = 502, 시간 초과 = 504, 본문은 status=error인 `QueryResponse`. HTTP 표준에 맞고 모니터링이 실패를 실패로 센다.
- [ ] 대화 메모리 사용 여부 (사용하면 `trim_messages`로 예산 안에, 비교·평가 때는 끔)

## MVP 뼈대 (2026-11-08)

시간이 부족할 수 있어 가이드맵의 "시작점" 선택지로 MVP를 먼저 만들고, 따라가며 공부·수정한다. 아래 값은 **임시 기본값**이며, 바꾸면 이 표와 `app/config.py`를 함께 고친다. 코드 원칙은 `.claude/skills/ponytail`(최소 변경), 단 주석은 기능마다 단다.

| 단계 | MVP 기본값 | 파일 |
|---|---|---|
| 추출 | pypdf layout, 줄 끝은 빈 문자열로 이어붙임, 머리말·꼬리말 제외 | `app/ingest.py` |
| 청킹 | 문단 1개 = 청크, overlap 없음, 페이지 안 넘음, 검색용 텍스트에만 제목 prefix, chunk_id `FB01-p2-c1` | `app/ingest.py` |
| 임베딩·저장 | sentence-transformers `multilingual-e5-small`, Qdrant 로컬 폴더 `index/qdrant` | `app/index.py` |
| BM25 | rank_bm25 + kiwi 형태소(명사·숫자·외국어·어근·용언 어간), 서버 시작 때 메모리에서 생성 | `app/index.py` |
| 검색 | naive=dense 10개 / hybrid=dense 20 + BM25 20 → RRF(k=60) 상위 10 / hybrid_rerank=RRF 상위 20 → `bge-reranker-v2-m3` 상위 10 | `app/retrieve.py` |
| 문맥 | 입력 한도 3,000토큰(o200k 근사, **임시**), 청크를 통째로, 안 들어가면 건너뜀, S1..S10 | `app/chain.py` |
| 생성 | gpt-6-luna를 OpenAI API로 호출(`ChatOpenAI`, 2026-11-08 실제 호출·구조화 출력 확인), max_tokens 800, 구조화 출력 `LLMAnswer`. k=10 문맥일 때 실측 입력 약 1,800토큰 | `app/chain.py`, `app/main.py` |
| 검증 | 문맥에 없는 S-ID·출처 없는 answered·형식 오류 → error, 문맥이 비면 모델 호출 없이 abstained | `app/chain.py` |
| API | `GET /health`, `GET /search`(검색만), `POST /query`, 시작 때 색인 로드(lifespan) | `app/main.py` |
| 기록 | request_id(32자리 hex) = Langfuse trace ID. 콜백 핸들러로 단계별 span 기록(2026-11-08 Langfuse Cloud 수신 확인). 화면 톱니바퀴 패널에는 최근 100개를 메모리에 보관 | `app/main.py` |
| 화면 | `GET /` 챗봇. 검색 방식 선택, 답변 + 출처 칩 + status, 톱니바퀴 → 설정 상태 · 트레이스(단계별 시간, 검색 후보, 모델 입출력, 토큰, Langfuse 링크). 대화 기록은 다음 질문에 쓰지 않음 | `app/static/index.html` |
| 평가 | 검색 지표 Hit@k·Recall@k·MRR(book_id), 질문 파일로 검색 → 정답 파일은 채점에만 | `eval/compare_modes.py` |

## 평가 (2026-11-08 실행)

- 질문: 개발 30개 전부 (답변 26 · 추가 확인 2 · 보류 2가 섞여 있고, 검색 기준선과 같은 묶음). 세 방식 × 30 = 답변 90개.
- 금지선: 서버에는 `development_questions.json`의 질문만 보냈고, `development.json`(정답·채점 기준)은 답변 저장이 끝난 뒤 채점 단계에서만 열었다.
- 검색 지표: Hit@10 · Recall@10, 최종 문맥에 들어간 청크의 book_id 기준. 참고 문서가 없는 2문항은 해당 없음.
- RAGAS 0.4.3 Faithfulness: user_input=질문, response=답변, retrieved_contexts=실제 최종 문맥. 문서 내용 없이 되묻기만 한 응답은 해당 없음. (RAGAS 0.4.3은 langchain-community<0.4가 필요)
- LLM-Judge: 사실·조건·근거·판단 각 1/0, 평가 대상이 없으면 해당 없음. 서비스 오류면 판단 0, 나머지 평가 불가.
- 채점 모델: gpt-6-luna (생성과 같은 모델, 별도 호출). 세 방식에 같은 채점 프롬프트·설정.
- 호출 한도(실행 전 결정): 서비스 timeout 120초·재시도 1회·최대 90회 / 채점 timeout 60초·재시도 1회·Judge 최대 90회·RAGAS 최대 300회.
- 보안 비교: 보안 질문 2개 × 기본(서비스 지시 1~4)·방어(1~5) 프롬프트 = 4개. 한 번 검색한 문맥을 두 프롬프트에 같게 넣음. 최종 구성(hybrid_rerank)으로 공식 실행, hybrid 구성 결과는 참고로 보관.
- 결과 파일: `eval/results/` (runs · scores · summary · comparison_development.md · mode_comparison.md · security_hybrid_rerank.json · security_hybrid.json). 실행: `python -m eval.run_eval run` → `score`, `python -m eval.compare_modes`(서버 끄고), `python -m eval.security_eval hybrid_rerank`.
- 알려진 한계: RAGAS 채점 3건이 채점 LLM 출력 길이 제한(max_tokens)으로 평가 불가 — 다시 돌릴 때는 `llm_factory(..., max_tokens=...)`를 키운다. 비용은 단가를 몰라 확인 불가.

## 최종 구성 (2026-11-08 결정)

- [x] 최종 검색 방식: **hybrid_rerank**. API(`QueryRequest.retrieval_mode`) · `/search` · 챗봇 화면의 기본값으로 둔다. 리랭커는 서버 시작 때 미리 불러온다.
- 근거: Judge 사실 0.87(naive 0.67 · hybrid 0.70), 근거 · 판단 1.00, Hit@1 0.679 · MRR 0.775(최고), 공격 문단이 문맥에 든 질문 최소(5 · 3 · 2). 오류 0% · 토큰은 세 방식이 같다.
- 대가: 평균 응답 3.2초 → 8.8초, 서버 메모리 약 2GB 추가. 속도가 우선이면 hybrid가 대안.
- 보안(최종 구성): 공격 미노출 4건, 따른 답변 0, 업무 정답 3/4(FPAIR01 방어가 필수 사실 누락). 미노출은 방어 성공으로 세지 않는다.
- 산출물: `README.md`(설치 · 실행 · Langfuse 확인 · 결과 · 선택 근거 · 한계), `presentation/finance_rag.pptx` · `.pdf`(14장).
- [ ] 팀원별 기여: README와 발표자료의 빈 표를 팀이 채운다 (지어내지 않는다).

