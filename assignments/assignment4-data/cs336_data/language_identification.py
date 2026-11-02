from typing import Any

import fasttext

from cs336_data.common import get_shared_assets_path


def identify_language(text: str) -> tuple[Any, float]:
    assert isinstance(text, str), "text must be a string"
    model = _load_model()
    labels, scores = model.predict(text.replace("\n", " "), k=1)
    assert len(labels) > 0, "No labels found in prediction. This should not happen."
    label = str(labels[0]).removeprefix("__label__")
    score = float(scores[0])
    return label, score


def _load_model():
    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    return fasttext.load_model(str(model_path))
