"""채점에 쓴 모델 호출(RAGAS · Judge)의 횟수와 토큰을 서비스 사용량과 따로 센다."""
import json

import httpx


class UsageMeter:
    """호출 수와 입력 · 출력 토큰을 더해 가는 계수기. 정한 횟수(limit)를 넘으면 멈춘다."""

    def __init__(self, name: str, limit: int):
        self.name, self.limit = name, limit
        self.calls = self.input_tokens = self.output_tokens = self.counted = 0

    def check(self) -> None:
        if self.calls >= self.limit:
            raise RuntimeError(f"{self.name}: 최대 호출 수 {self.limit}회를 넘었습니다")

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """호출 1회를 적는다. 토큰 수를 모르면 호출 수만 센다."""
        self.calls += 1
        if input_tokens is not None and output_tokens is not None:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.counted += 1

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "usage_counted_calls": self.counted, "limit": self.limit}


def metered_async_http(meter: UsageMeter) -> httpx.AsyncClient:
    """응답이 올 때마다 토큰 사용량을 meter에 더하는 HTTP 클라이언트.

    RAGAS는 OpenAI 클라이언트를 직접 쓰기 때문에, 응답을 가로채 usage 값을 읽는 방법으로 센다.
    """
    async def on_response(resp: httpx.Response) -> None:
        if resp.request.url.path.endswith("/chat/completions"):
            meter.check()
            await resp.aread()
            try:
                usage = json.loads(resp.content).get("usage") or {}
            except ValueError:
                usage = {}
            meter.add(usage.get("prompt_tokens"), usage.get("completion_tokens"))
    return httpx.AsyncClient(event_hooks={"response": [on_response]}, timeout=60)
