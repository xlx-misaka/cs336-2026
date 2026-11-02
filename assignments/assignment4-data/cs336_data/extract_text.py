from cs336_data.common import get_shared_assets_path


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


def _extract_text_from_html_bytes_example():
    """Example usage of extract_text_from_html_bytes."""
    from fastwarc.warc import ArchiveIterator, WarcRecordType

    data_path = get_shared_assets_path() / "CC" / "example.warc.gz"

    with open(data_path, "rb") as f:
        for index, record in enumerate(ArchiveIterator(f)):
            if index == 100:
                break
            if record.record_type == WarcRecordType.response and record.http_content_type == "text/html":
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                if text is not None:
                    print(f"Extracted text from record {index}:\n{text[:100]}\n...\n")
                else:
                    print(f"Failed to extract text from record {index}.")


if __name__ == "__main__":
    _extract_text_from_html_bytes_example()
