def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Decode raw HTML and extract its visible text with Resiliparse."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    if not isinstance(html_bytes, bytes):
        raise TypeError("html_bytes must be bytes")

    try:
        try:
            html = html_bytes.decode("utf-8")
        except UnicodeDecodeError:
            encoding = detect_encoding(html_bytes) or "utf-8"
            html = html_bytes.decode(encoding, errors="replace")
        return extract_plain_text(html)
    except (LookupError, UnicodeError, ValueError):
        return None
