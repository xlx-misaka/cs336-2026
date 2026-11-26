from typing import Any

import regex as re


def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    response = re.split(
        r"^[ \t]*#\s*Query\s*:",
        model_output,
        maxsplit=1,
        flags=re.I | re.M,
    )[0]
    response = response.replace("**", "").replace("`", "").strip()

    answer_marker = r"answer(?:\s+is\s+|\s*:\s*)"
    qualified_prefix = r"\b(?:(correct|final|wrong|incorrect)\s+)?" + answer_marker
    option_pattern = (
        r"(?:option\s+)?\(?([A-D])"
        r"(?:\)|(?=[ \t]*(?:[.!?,:;\n]|$)|[ \t]+because\b))"
    )
    answer_matches = re.findall(
        qualified_prefix + option_pattern,
        response,
        flags=re.I,
    )

    final_letters = {
        letter.upper()
        for qualifier, letter in answer_matches
        if qualifier.casefold() == "final"
    }
    if final_letters:
        return next(iter(final_letters)) if len(final_letters) == 1 else None

    letters = {
        letter.upper()
        for qualifier, letter in answer_matches
        if qualifier.casefold() not in {"wrong", "incorrect"}
    }
    if letters:
        return next(iter(letters)) if len(letters) == 1 else None

    leading_option = re.match(r"^\(?([A-D])(?:[.)](?=\s|$)|$)", response, re.I)
    if leading_option:
        return leading_option.group(1).upper()

    answer_prefix = r"\b(?:(?:correct|final)\s+)?" + answer_marker
    answer_text = re.sub(r"^(?:the\s+)?" + answer_prefix, "", response, flags=re.I)
    normalized = " ".join(answer_text.casefold().rstrip(" .!?").split())

    matches = [
        letter
        for letter, option in zip("ABCD", mmlu_example["options"])
        if " ".join(option.casefold().rstrip(" .!?").split()) == normalized
    ]

    return matches[0] if len(matches) == 1 else None


def parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    number_pattern = (
        r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
    )
    numbers = re.findall(number_pattern, model_output.replace("\u2212", "-"))
    if not numbers:
        return None

    return numbers[-1].replace(",", "").removeprefix("+")
