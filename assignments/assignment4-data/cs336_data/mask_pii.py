import regex as re

from cs336_data.common import get_shared_assets_path

_MASK_EMAILS_RE = re.compile(
    r"(?<![A-Za-z0-9!#$%&'*+/=?^_`{|}~.@-])"
    r"(?=[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}"
    r"(?![A-Za-z0-9_@-]|\.+[A-Za-z0-9_-])"
    r"(?<![A-Za-z0-9!#$%&'*+/=?^_`{|}~.@-]{255})"
)

_MASK_PHONES_RE = re.compile(
    r"""
    # Skip complete DOI tokens, including their phone-like numeric suffixes.
    \b10\.[0-9]{4,9}/[^\s<>"]+(*SKIP)(*FAIL)
    |
    (?<![\w+])
    (?:
        # UK domestic and international numbers: 11 or 10 national digits.
        (?:(?:\+44|0044)[ \t.-]*(?:\(0\)[ \t.-]*)?|0)
        [1235789](?:[ \t.-]*[0-9]){9}
        |
        # Common UK area codes enclosed in parentheses.
        \(0[1235789][0-9]{3}\)(?:[ \t.-]*[0-9]){6}
        |
        \(0[1235789][0-9]{2}\)(?:[ \t.-]*[0-9]){7}
        |
        \(0[12][0-9]\)(?:[ \t.-]*[0-9]){8}
        |
        # Existing North American full and local number formats.
        (?:\+?1[ \t.-]*)?
        (?:\([ \t]*[0-9]{3}[ \t]*\)|[0-9]{3})
        [ \t.-]*[0-9]{3}[ \t.-]*[0-9]{4}
        |
        [0-9]{3}[ \t.-]+[0-9]{4}
    )
    (?:[ \t]*,?[ \t]*(?:ext(?:ension)?\.?|x|\#)[ \t:]*[0-9]+)?
    (?!\w)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MASK_IPS_RE = re.compile(
    r"(?<![0-9])(?<![0-9]\.)"
    r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
    r"(?![0-9]|\.[0-9])"
)


def mask_emails(text: str) -> tuple[str, int]:
    """Mask all email addresses in the text."""
    return _MASK_EMAILS_RE.subn("|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Mask common US/UK phone formats while preserving DOI tokens."""
    return _MASK_PHONES_RE.subn("|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    """Mask all IP addresses in the text."""
    return _MASK_IPS_RE.subn("|||IP_ADDRESS|||", text)


def _mask_pii_example():
    """Example mask pii."""
    from fastwarc.warc import ArchiveIterator, WarcRecordType

    from cs336_data.extract_text import extract_text_from_html_bytes

    data_path = get_shared_assets_path() / "CC" / "example.warc.gz"

    with open(data_path, "rb") as f:
        for index, record in enumerate(ArchiveIterator(f)):
            if index == 100:
                break
            if record.record_type == WarcRecordType.response and record.http_content_type == "text/html":
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                if text is not None:
                    print(f"Extracted text from record {index}:\n{text[:100]}\n...")
                    emails = _MASK_EMAILS_RE.findall(text)
                    phone_numbers = _MASK_PHONES_RE.findall(text)
                    ips = _MASK_IPS_RE.findall(text)
                    print(emails)
                    print(phone_numbers)
                    print(ips)
                else:
                    print(f"Failed to extract text from record {index}.\n")


if __name__ == "__main__":
    _mask_pii_example()
