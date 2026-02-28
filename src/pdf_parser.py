"""
pdf_parser.py
-------------
Handles PDF ingestion and text extraction using PyMuPDF.
Supports multi-page PDFs with robust error handling and text normalization.
"""

import re
import io
import logging
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract raw text from PDF bytes.

    Args:
        pdf_bytes: Raw PDF file content as bytes.

    Returns:
        Concatenated plain text from all pages.

    Raises:
        ValueError: If the PDF cannot be opened or has no extractable text.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.page_count == 0:
        raise ValueError("PDF contains no pages.")

    pages_text: list[str] = []
    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        except Exception as exc:
            logger.warning("Could not extract text from page %d: %s", page_num, exc)

    doc.close()

    if not pages_text:
        raise ValueError("No extractable text found in the PDF.")

    return "\n".join(pages_text)


def normalize_text(raw_text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    Steps:
        - Replace line breaks / carriage returns with spaces
        - Collapse multiple whitespace characters
        - Remove non-ASCII noise (optional, preserves common punctuation)
        - Strip leading/trailing whitespace

    Args:
        raw_text: Unprocessed text string.

    Returns:
        Normalized text string.
    """
    # Replace various newline/tab chars with a space
    text = raw_text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Remove bullet / special unicode chars that add noise
    text = re.sub(r"[•●▪▸►◆■□✓✔✗✘]", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s{2,}", " ", text)

    # Strip
    text = text.strip()

    return text


def parse_pdf(pdf_bytes: bytes) -> dict:
    """
    Full pipeline: extract and normalize text from a PDF byte stream.

    Args:
        pdf_bytes: Raw PDF file content as bytes.

    Returns:
        Dictionary with keys:
            - 'raw_text': Unprocessed extracted text.
            - 'clean_text': Normalized text ready for NLP.
            - 'page_count': Number of pages in document.
    """
    raw_text = extract_text_from_bytes(pdf_bytes)
    clean_text = normalize_text(raw_text)

    # Determine page count (reopen briefly)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count
        doc.close()
    except Exception:
        page_count = -1

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "page_count": page_count,
    }
