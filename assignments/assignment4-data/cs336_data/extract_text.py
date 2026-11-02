import resiliparse.extract.html2text
import resiliparse.parse


def extract_text(html_bytes: bytes):
    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding=encoding)
    html_text = resiliparse.extract.html2text.extract_plain_text(html_str)
    return html_text
