from pathlib import Path
from pypdf import PdfReader

def load_txt(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def load_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({"text": text, "page": i + 1})

    return pages
