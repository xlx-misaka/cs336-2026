from nltk.tokenize import word_tokenize

_SEED = 336
_MIN_WORDS = 50
_MAX_WORDS = 100_000
_MIN_MEAN_WORD_LENGTH = 3.0
_MAX_MEAN_WORD_LENGTH = 10.0
_MAX_ELLIPSIS_LINE_RATIO = 0.30
_MIN_ALPHABETIC_WORD_RATIO = 0.80


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


class QualityFilter: ...
