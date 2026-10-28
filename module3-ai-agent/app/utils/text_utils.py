import re
import unicodedata


def sanitize_filename(filename: str) -> str:
    filename = filename.replace("\\", "/").split("/")[-1]
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")
    filename = re.sub(r"[^\w\-.]", "_", filename)
    filename = filename.lstrip(".")
    if len(filename) > 200:
        name, _, ext = filename.rpartition(".")
        filename = name[:196] + "." + ext if ext else filename[:200]
    return filename or "document"


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and extra blank lines; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Normalise PDF/extracted text without changing meaning, spelling, or language.

    Soft line-wraps (single newlines) become spaces. Paragraph breaks (blank
    lines) are preserved as a single blank line.
    """
    if not text:
        return ""
    text = normalize_whitespace(text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
