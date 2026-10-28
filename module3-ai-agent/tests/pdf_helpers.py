from pathlib import Path

import fitz


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Create a small text PDF for tests. One string per page."""
    document = fitz.open()
    try:
        for text in pages:
            page = document.new_page()
            if text:
                page.insert_textbox(fitz.Rect(72, 72, 520, 770), text)
        document.save(path)
    finally:
        document.close()
    return path


def long_text(word_count: int, prefix: str = "term") -> str:
    return " ".join(f"{prefix}{i}" for i in range(word_count))
