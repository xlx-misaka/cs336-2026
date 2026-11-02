from typing import Any

import fasttext

from cs336_data.common import get_shared_assets_path

_NSFW_CLASSIFIER = None
_TOXIC_CLASSIFIER = None


def classify_nsfw(text: str) -> tuple[Any, float]:
    """Classify text as NSFW or not NSFW."""
    classifier = _load_nsfw_classifier()
    labels, scores = classifier.predict(text, k=1)
    assert len(labels) > 0
    label = str(labels[0]).removeprefix("__label__")
    score = float(scores[0])
    return label, score


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    """Classify text as toxic or not toxic."""
    classifier = _load_toxic_classifier()
    labels, scores = classifier.predict(text, k=1)
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
