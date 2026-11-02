from nltk.tokenize import word_tokenize


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
        50 <= num_words <= 100000
        and 3 <= mean_word_length <= 10
        and ellipsis_lines_percentage <= 0.3
        and alphabetic_word_percentage >= 0.8
    )


class QualityFilter:
    def __init__(self, min_lines: int = 30, max_lines: int = 100000):
        self.min_lines = min_lines
        self.max_lines = max_lines
