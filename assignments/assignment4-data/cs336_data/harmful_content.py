from typing import Any

import fasttext

from cs336_data.common import get_shared_assets_path

_NSFW_CLASSIFIER = None
_TOXIC_CLASSIFIER = None


def classify_nsfw(text: str) -> tuple[Any, float]:
    """Classify text as NSFW or not NSFW."""
    classifier = _load_nsfw_classifier()
    labels, scores = classifier.predict(text.replace("\n", " "), k=1)
    assert len(labels) > 0
    label = str(labels[0]).removeprefix("__label__")
    score = float(scores[0])
    return label, score


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    """Classify text as toxic or not toxic."""
    classifier = _load_toxic_classifier()
    labels, scores = classifier.predict(text.replace("\n", " "), k=1)
    assert len(labels) > 0
    label = str(labels[0]).removeprefix("__label__")
    score = float(scores[0])
    return label, score


def _load_nsfw_classifier():
    global _NSFW_CLASSIFIER

    if _NSFW_CLASSIFIER is not None:
        return _NSFW_CLASSIFIER

    nsfw_path = get_shared_assets_path() / "classifiers" / "dolma_fasttext_nsfw_jigsaw_model.bin"
    _NSFW_CLASSIFIER = fasttext.load_model(str(nsfw_path))

    return _NSFW_CLASSIFIER


def _load_toxic_classifier():
    global _TOXIC_CLASSIFIER

    if _TOXIC_CLASSIFIER is not None:
        return _TOXIC_CLASSIFIER

    toxic_path = get_shared_assets_path() / "classifiers" / "dolma_fasttext_hatespeech_jigsaw_model.bin"
    _TOXIC_CLASSIFIER = fasttext.load_model(str(toxic_path))

    return _TOXIC_CLASSIFIER


def _classify_harmful_content_example():
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
                    label, score = classify_nsfw(text)
                    print(f"classify nsfw: {label}, score: {score:.2f}")
                    label, score = classify_toxic_speech(text)
                    print(f"classify toxic speech: {label}, score: {score:.2f}\n")
                else:
                    print(f"Failed to extract text from record {index}.\n")


if __name__ == "__main__":
    _classify_harmful_content_example()
