"""PDF를 읽어 문단 단위 청크로 나눈다.

규칙
- 문단 1개 = 청크 1개. 페이지를 넘지 않는다.
- overlap: 같은 페이지 앞 문단의 마지막 1문장을 검색용 글(search_text)에 함께 넣는다.
  답변 근거로 보여 주는 원문(text)에는 넣지 않아 인용은 그 문단 그대로 남는다.
- 페이지 맨 위 머리말(`FB01 | 부서`)과 맨 아래 쪽번호(`n / 24`)는 뺀다.
- 검색에 쓰는 글(search_text)에만 "문서 제목 > 소제목"을 앞에 붙인다.
"""
import re
from pathlib import Path

from pypdf import PdfReader

from app.config import DOC_DIR

# 문장 끝: 마침표 · 물음표 · 느낌표 뒤 (닫는 따옴표가 있으면 그 뒤). "3.5"처럼 뒤에 숫자가 오면 자르지 않는다.
SENTENCE_END = re.compile(r'(?<=[.?!][”"])\s*|(?<=[.?!])(?![”"\d])\s*')


def split_paragraphs(page_text: str) -> list[str]:
    """빈 줄을 기준으로 페이지 글을 문단으로 나눈다.

    PDF는 한 문단을 여러 줄로 끊어 두므로 줄과 줄을 빈칸 없이 붙인다 ("매출" + "은" → "매출은").
    """
    paragraphs, lines = [], []
    for line in page_text.splitlines():
        if line.strip():
            lines.append(line.strip())
        elif lines:
            paragraphs.append("".join(lines))
            lines = []
    if lines:
        paragraphs.append("".join(lines))
    return paragraphs


def last_sentence(paragraph: str) -> str:
    """문단의 마지막 문장 ("가다. 나다. 다다." → "다다.")."""
    return [s for s in SENTENCE_END.split(paragraph) if s.strip()][-1]


def load_chunks(doc_dir: Path = DOC_DIR) -> list[dict]:
    """모든 PDF를 청크 목록으로 만든다.

    청크 예: {"chunk_id": "FB01-p2-c2", "book_id": "FB01", "page": 2, "text": 원문,
             "search_text": 제목 + 앞 문단 마지막 문장 + 원문, ...}
    """
    chunks = []
    for pdf in sorted(doc_dir.glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            parts = split_paragraphs(page.extract_text(extraction_mode="layout"))
            # 모든 페이지가 [머리말, 제목, 소제목, 문단..., 쪽번호] 모양이었다. 아니면 조용히 넘기지 않고 멈춘다.
            if len(parts) < 5:
                raise ValueError(f"{pdf.name} {page_no}쪽의 모양이 예상과 다릅니다 (덩어리 {len(parts)}개)")
            _header, title, heading, *body, _page_number = parts
            for n, text in enumerate(body, start=1):
                # 페이지 첫 문단은 같은 페이지에 앞 문단이 없어 겹칠 문장이 없다
                overlap = last_sentence(body[n - 2]) + " " if n > 1 else ""
                chunks.append({
                    "chunk_id": f"{pdf.stem}-p{page_no}-c{n}",
                    "book_id": pdf.stem,
                    "file_name": pdf.name,
                    "page": page_no,
                    "title": title,
                    "heading": heading,
                    "text": text,
                    "search_text": f"{title} > {heading}\n{overlap}{text}",
                })
    return chunks
