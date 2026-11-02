from typing import Any

_MODEL: Any | None = None


def identify_language(text: str) -> tuple[Any, float]:

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text must not be empty")

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

    import fasttext

    from cs336_data.common import get_shared_assets_path

    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    _MODEL = fasttext.load_model(str(model_path))

    return _MODEL
