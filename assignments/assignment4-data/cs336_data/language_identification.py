from typing import Any

import fasttext

from cs336_data.common import get_shared_assets_path

_MODEL = None


def identify_language(text: str) -> tuple[Any, float]:
    """Identify the language of the given text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return "empty", 1.0

    model = _load_model()
    labels, scores = model.predict(text.replace("\n", " "), k=1)

    assert len(labels) > 0

    label = str(labels[0]).removeprefix("__label__")
    score = float(scores[0])
    return label, score


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    _MODEL = fasttext.load_model(str(model_path))

    return _MODEL


def _identify_language_example():
    """Example usage of the language identification model."""
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
                    lang, score = identify_language(text)
                    print(f"Identified language: {lang}, score: {score:.2f}\n")
                else:
                    print(f"Failed to extract text from record {index}.\n")


if __name__ == "__main__":
    _identify_language_example()
