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
    (?<![\w+./-])
    (?:
        # Explicit international prefixes; bounded length avoids matching long IDs.
        (?:\+|00)[1-9](?:[ \t().-]*[0-9]){7,14}
        |
        # UK domestic numbers and common Australian domestic formats.
        0[1235789](?:[ \t.-]*[0-9]){9}
        |
        0[23478](?:[ \t.-]*[0-9]){8}
        |
        \(0[2378]\)(?:[ \t.-]*[0-9]){8}
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
        (?P<local>[0-9]{3}[ \t.-]+[0-9]{4})
    )
    (?:-[0-9]{1,4})?
    (?:[ \t]*,?[ \t]*(?:ext(?:ension)?\.?|x|\#)[ \t:]*[0-9]+)?
    (?![\w/-]|\.[0-9])
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Seven-digit local numbers also resemble postcodes; require a nearby phone cue.
_PHONE_CONTEXT_RE = re.compile(
    r"\b(?:tel(?:ephone)?|phone|fax|mobile|call|contact)\b"
    r"(?:[ \t]+(?:number|us|at)){0,2}[ \t\r\n:(]*\Z",
    re.IGNORECASE,
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
    """Mask common phone formats; require context for ambiguous seven-digit numbers."""
    count = 0

    def replace(match):
        nonlocal count
        if match.group("local") and not _PHONE_CONTEXT_RE.search(text[max(0, match.start() - 80) : match.start()]):
            return match.group()
        count += 1
        return "|||PHONE_NUMBER|||"

    return _MASK_PHONES_RE.sub(replace, text), count


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
                    _, phone_numbers_masked = mask_phone_numbers(text)
                    ips = _MASK_IPS_RE.findall(text)
                    print(emails)
                    print(f"Phone numbers masked: {phone_numbers_masked}")
                    print(ips)
                else:
                    print(f"Failed to extract text from record {index}.\n")


if __name__ == "__main__":
    _mask_pii_example()
