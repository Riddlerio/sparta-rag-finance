"""설정값 모음. 세 검색 방식을 비교할 때는 여기 값을 바꾸지 않는다."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env 파일의 API 키를 읽어 온다 (.env는 저장소에 올리지 않는다)

# 폴더와 파일
ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = ROOT / "data" / "documents"     # 검색 대상 PDF. 평가용 JSON은 절대 넣지 않는다
EVAL_DIR = ROOT / "data" / "evaluation"
INDEX_DIR = ROOT / "index"                # 색인 결과를 저장하는 곳
CHUNKS_PATH = INDEX_DIR / "chunks.json"   # 청크 원문과 위치 정보
QDRANT_PATH = INDEX_DIR / "qdrant"        # 벡터 저장소 (내 PC 폴더에 저장)
COLLECTION = "finance_chunks"

# 모델
EMBED_MODEL = "intfloat/multilingual-e5-small"    # 문장 → 벡터 (의미 검색용)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"          # 질문과 청크를 같이 읽고 점수를 매기는 모델
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-6-luna")  # 답변 생성 모델

# 검색과 문맥
TOP_K = 10          # 모델에게 보여 줄 청크 수
CANDIDATES = 20     # 하이브리드에서 의미 · 키워드 검색이 각각 가져오는 후보 수
RRF_K = 60          # 순위를 합칠 때 쓰는 상수: 점수 = 1 / (60 + 순위)

# 모델 입력 길이. 3,000은 모델의 실제 한도를 확인하지 못해 정한 넉넉한 임시값이다.
INPUT_TOKEN_LIMIT = 3000   # 지시문 + 질문 + 문맥을 합친 최대 토큰
MAX_ANSWER_TOKENS = 800    # 답변 최대 토큰
