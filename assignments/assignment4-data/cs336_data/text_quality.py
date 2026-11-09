"""Conservative text-only signals for the corpus pipeline, separate from course adapters.

These heuristics provide evidence for reviewing classifier rejections; they are not
calibrated probabilities or a substitute for validation perplexity measurements.
"""

import re
from dataclasses import dataclass
from io import StringIO

_FUNCTION_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "that",
        "as",
        "it",
        "you",
        "on",
        "are",
        "be",
        "we",
        "this",
        "was",
        "from",
        "by",
        "is",
        "have",
        "has",
        "can",
        "will",
        "would",
        "should",
        "their",
        "they",
        "its",
        "his",
        "her",
        "our",
        "your",
        "not",
        "through",
        "about",
        "into",
        "over",
        "under",
        "between",
        "after",
        "before",
        "than",
        "then",
        "who",
        "which",
        "what",
        "when",
        "where",
        "how",
        "why",
        "without",
        "during",
        "within",
        "these",
        "those",
        "each",
        "both",
    ]
)
_AUXILIARIES = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "do",
        "does",
        "did",
    ]
)
_CODE_WORDS = frozenset(
    [
        "def",
        "class",
        "return",
        "import",
        "function",
        "const",
        "let",
        "var",
        "public",
        "if",
        "for",
        "while",
        "select",
        "lambda",
        "defun",
        "defmacro",
    ]
)
_NAVIGATION = frozenset(
    [
        "home",
        "about",
        "news",
        "services",
        "contact",
        "menu",
        "main menu",
        "menu button",
        "search",
        "search:",
        "sign in",
        "log in",
        "login",
        "sign up",
        "register",
        "subscribe",
        "privacy",
        "privacy policy",
        "cookie policy",
        "terms",
        "terms of use",
        "terms and conditions",
        "contact us",
        "about us",
        "follow us",
        "share",
        "skip to content",
        "skip to main content",
        "toggle navigation",
        "remember me",
        "password",
        "username or e-mail",
        "forgot your password?",
    ]
)
_CODE_START = re.compile(
    r"^\s*(?:\((?:defun|defmacro|lambda)\b|(?:def|class|function|return|import|from|const|let|var)\b"
    r"|[(){}\[\]]+\s*$|//|;;)|^(?: {4}|\t).*?[(){}=;]"
)
_FOOTER = re.compile(
    r"^(?:copyright\b|©|all rights reserved\b|powered by\b|built by\b|website design\b|site build\b)", re.IGNORECASE
)
_TEASER = re.compile(r"(?:\.{3}|…)(?:\s*(?:read more|continue reading|\[.*?\]))?\s*$", re.IGNORECASE)
_PARKED = re.compile(
    r"does not have an active website|domain (?:is )?(?:parked|for sale)|website (?:is )?coming soon"
    r"|^this domain is (?:now )?available (?:at|for) auction[.! ]",
    re.IGNORECASE | re.MULTILINE,
)
_PAYWALL = re.compile(
    r"subscribers? only|subscription required|members? only|(?:sign|log) in to (?:continue|read|access)"
    r"|subscribe.{0,60}(?:full (?:article|story)|continue reading)",
    re.IGNORECASE,
)
_EMPTY_SEARCH = re.compile(
    r"no (?:search )?results (?:found|for)|quoted phrase not found in phrase index", re.IGNORECASE
)
_INDEX_PAGE = re.compile(r"\b(?:archives?|search results|posts tagged|author at|page \d+)\b", re.IGNORECASE)
_UNAVAILABLE_JOB = re.compile(
    r"^(?:sorry[, ]+(?:but )?)?(?:this|the) (?:job|position|listing) (?:is no longer available|has expired)[.! ]",
    re.IGNORECASE | re.MULTILINE,
)
_COOKIE_CONTEXT = re.compile(
    r"cookies? (?:settings|notice|policy|consent)|privacy\s*&\s*cookies|accept all|save\s*&\s*accept|consent plugin",
    re.IGNORECASE,
)
_COOKIE_TEMPLATE = re.compile(
    r"^(?:this (?:website|site) uses cookies|we use cookies (?:to|on)|we also use third-party cookies|these cookies"
    r"|necessary cookies|any cookies that|other uncategorized cookies|cookielawinfo[-_]|the cookie is set"
    r"|the cookies (?:is|are) used|(?:analytical|advertisement|performance|functional|non-necessary) cookies)"
    r"|GDPR Cookie Consent plugin|consent prior to running these cookies",
    re.IGNORECASE,
)


def is_code_line(line: str) -> bool:
    return _CODE_START.search(line) is not None


def prose_word_count(line: str) -> int:
    words = line.split()
    if len(words) < 8 or is_code_line(line) or _TEASER.search(line):
        return 0
    alphabetic = sum(any(char.isalpha() for char in word) for word in words)
    tokens = set(re.findall(r"[a-z]+", line.lower()))
    if alphabetic < 0.7 * len(words) or len(tokens & _FUNCTION_WORDS) < 2:
        return 0
    if not (re.search(r"[.!?,;]", line) or tokens & _AUXILIARIES):
        return 0
    return len(words)


@dataclass(frozen=True)
class TextSignals:
    words: int
    prose_words: int
    complete_prose_words: int
    code_lines: int
    code_keywords: int
    navigation_lines: int
    teaser_lines: int
    teaser_words: int
    mean_alpha_length: float

    @property
    def readable_short(self) -> bool:
        return (
            12 <= self.words < 50
            and self.complete_prose_words >= 12
            and self.prose_words >= 0.4 * self.words
            and 3 <= self.mean_alpha_length <= 10
        )

    @property
    def substantial_body(self) -> bool:
        prose = self.prose_words >= 120 and self.prose_words >= 0.35 * self.words and 3 <= self.mean_alpha_length <= 10
        code = self.code_lines >= 3 and (self.prose_words >= 30 or self.code_keywords >= 3)
        return self.words <= 100_000 and (prose or code)


def text_signals(text: str) -> TextSignals:
    words = prose = complete = code = navigation = teasers = teaser_words = 0
    alpha_length = alpha_count = 0
    keywords = set()
    for line in text.splitlines():
        tokens = line.split()
        words += len(tokens)
        navigation += int(line.strip().lower() in _NAVIGATION)
        if is_code_line(line):
            code += 1
            keywords.update(set(re.findall(r"[a-z]+", line.lower())) & _CODE_WORDS)
        if _TEASER.search(line):
            teasers += 1
            teaser_words += len(tokens)
        count = prose_word_count(line)
        prose += count
        if count:
            alpha = re.findall(r"[^\W\d_]+", line)
            alpha_length += sum(map(len, alpha))
            alpha_count += len(alpha)
            if re.search(r"[.!?][\"'’)]?\s*$", line):
                complete += count
    return TextSignals(
        words,
        prose,
        complete,
        code,
        len(keywords),
        navigation,
        teasers,
        teaser_words,
        alpha_length / alpha_count if alpha_count else 0,
    )


def clean_boilerplate(text: str) -> tuple[str, int]:
    """Remove standalone interface/footer lines, preserving paragraph breaks and code."""
    lines = []
    removed = 0
    title = next((line for line in text.splitlines() if line.strip()), "")
    cookie_ui = _COOKIE_CONTEXT.search(text) and not re.search(r"\bcookies?\b", title, re.IGNORECASE)
    for line in StringIO(text, newline=None):
        stripped = line.strip()
        navigation = stripped.lower() in _NAVIGATION
        footer = _FOOTER.search(stripped) and len(stripped.split()) <= 30 and not prose_word_count(stripped)
        cookie = cookie_ui and _COOKIE_TEMPLATE.search(stripped)
        if not is_code_line(line) and (navigation or footer or cookie):
            removed += 1
        else:
            lines.append(line)
    return "".join(lines).strip(), removed


def page_rejection_reason(text: str, signals: TextSignals, removed_navigation_lines: int = 0) -> str | None:
    if not text.strip():
        return "empty"
    if _UNAVAILABLE_JOB.search(text[:2000]):
        return "unavailable_job"
    if signals.substantial_body:
        return None
    if _PARKED.search(text):
        return "parked_domain"
    if _PAYWALL.search(text):
        return "paywall"
    if _EMPTY_SEARCH.search(text):
        return "empty_search"
    if signals.readable_short:
        return None
    if (
        _INDEX_PAGE.search(text[:1500])
        and signals.teaser_lines >= 2
        and signals.teaser_words >= max(40, 0.2 * signals.words)
    ):
        return "teaser_index"
    navigation = signals.navigation_lines + removed_navigation_lines
    if navigation >= 5 and signals.words >= 150 and signals.prose_words == 0 and signals.code_lines < 2:
        return "navigation"
    return None
