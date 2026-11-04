from typing import Any

import fasttext
from nltk.tokenize import word_tokenize

_MIN_WORDS = 50
_MAX_WORDS = 100_000
_MIN_MEAN_WORD_LENGTH = 3.0
_MAX_MEAN_WORD_LENGTH = 10.0
_MAX_ELLIPSIS_LINE_RATIO = 0.30
_MIN_ALPHABETIC_WORD_RATIO = 0.80

_QUALITY_CLASSIFIER = None


def gopher_quality_filter(text: str) -> bool:
    """
    A function that filters out low-quality text based on the following criteria:\n
    • Contain less than 50 or more than 100,000 words.\n
    • Have a mean word length outside the range of 3 to 10 characters.\n
    • Have more than 30% of lines ending with an ellipsis (“...”).\n
    • Contain less than 80% of words with at least one alphabetic character.
    """
    words = word_tokenize(text)
    num_words = len(words)

    mean_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0

    num_lines = len(text.split("\n"))
    ellipsis_lines_percentage = sum(line.endswith("...") for line in text.split("\n")) / num_lines

    alphabetic_words = sum(any(char.isalpha() for char in word) for word in words)
    alphabetic_word_percentage = alphabetic_words / num_words if num_words > 0 else 0

    return (
        _MIN_WORDS <= num_words <= _MAX_WORDS
        and _MIN_MEAN_WORD_LENGTH <= mean_word_length <= _MAX_MEAN_WORD_LENGTH
        and ellipsis_lines_percentage <= _MAX_ELLIPSIS_LINE_RATIO
        and alphabetic_word_percentage >= _MIN_ALPHABETIC_WORD_RATIO
    )


def classify_quality(text: str) -> tuple[Any, float]:
    classifier = _load_quality_classifier()
    labels, scores = classifier.predict(text.replace("\n", " "), k=1)
    assert len(labels) > 0
    label = str(labels[0]).removeprefix("__label__")
    score = float(scores[0])
    src = "cc" if label == "low-quality" else "wiki"
    return src, score


def _load_quality_classifier():
    global _QUALITY_CLASSIFIER
    if _QUALITY_CLASSIFIER is not None:
        return _QUALITY_CLASSIFIER

    path = "data/quality_classifier_data/quality_classifier.bin"
    _QUALITY_CLASSIFIER = fasttext.load_model(path)
    return _QUALITY_CLASSIFIER
