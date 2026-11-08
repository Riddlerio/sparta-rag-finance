"""데이터 형식 정의. 들어오는 요청, 모델의 답, 나가는 응답을 각각 검사한다."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Mode = Literal["naive", "hybrid", "hybrid_rerank"]
Status = Literal["answered", "needs_clarification", "abstained", "error"]


class QueryRequest(BaseModel):
    """POST /query 요청. 형식이 틀리면 FastAPI가 422로 돌려보낸다."""
    question: str = Field(min_length=1, max_length=1000, description="사용자 질문")
    retrieval_mode: Mode = Field("hybrid_rerank", description="검색 방식 (기본값은 최종 구성)")

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """공백만 있는 질문도 빈 질문으로 막는다."""
        if not v.strip():
            raise ValueError("질문이 비어 있습니다")
        return v.strip()


# 아래 설명 글은 모델에게 그대로 전달되는 답변 양식이다. 바꾸면 답변이 달라질 수 있다.
class LLMAnswer(BaseModel):
    """모델이 돌려줄 JSON 형식. error는 모델이 아니라 코드가 정하므로 여기 없다."""
    status: Literal["answered", "needs_clarification", "abstained"] = Field(
        description="answered=자료로 답함, needs_clarification=질문 대상이 모호함, abstained=자료에 근거가 없음")
    answer: str = Field(description="답변 또는 추가 확인 요청 · 보류 이유")
    source_ids: list[str] = Field(description="답변 근거로 쓴 문맥 라벨 (예: ['S1', 'S3'])")


class Source(BaseModel):
    """출처 하나: 모델이 쓴 라벨(S1 등)을 실제 문서와 쪽으로 바꾼 것."""
    source_id: str
    book_id: str
    file_name: str
    page: int
    chunk_id: str


class QueryResponse(BaseModel):
    """POST /query 응답."""
    request_id: str
    retrieval_mode: Mode
    status: Status
    answer: str
    sources: list[Source]
