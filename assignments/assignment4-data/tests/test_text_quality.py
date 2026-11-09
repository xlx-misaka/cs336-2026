from collections import Counter

import pytest

from cs336_data.text_quality import clean_boilerplate, page_rejection_reason, text_signals
from scripts import filter_data as pipeline

ARTICLE = (
    "This article explains how the system works and why the design matters. "
    "The examples show how readers can use the information in their own projects. "
) * 8
SHORT = "Maintenance report\nScheduled server maintenance is complete, and all services are available again today."


def write_shard(path, texts):
    import json

    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))


def test_complete_short_prose_has_evidence_but_menu_does_not():
    assert text_signals(SHORT).readable_short
    assert not text_signals("Home\nMenu\nSign in\nChoose your country\nTerms and conditions").readable_short


def test_complete_short_prose_can_include_title_date_and_navigation():
    text = (
        "Service maintenance and availability report for customers\nAbout\nNews\nServices\nContact\n"
        "18 November 2026\nScheduled server maintenance is complete, and all services are available again today."
    )
    cleaned, _ = clean_boilerplate(text)
    assert text_signals(cleaned).readable_short


def test_footer_cleanup_preserves_indented_code_and_article_lines():
    text = "Home\n" + ARTICLE + "\n\n    return None\n    return None\nPrivacy Policy\n"
    cleaned, removed = clean_boilerplate(text)
    assert removed == 2
    assert ARTICLE in cleaned
    assert "\n\n    return None\n    return None" in cleaned


@pytest.mark.parametrize(
    "message",
    [
        "Subscribers Only",
        "This domain does not have an active website",
        "This domain is now available at auction. Place your bid to make it yours.",
        "No search results found",
    ],
)
def test_explicit_empty_page_states_are_rejected(message):
    text = message + "\nHome\nMenu\nPlease use the form below to access the service."
    cleaned, removed = clean_boilerplate(text)
    assert page_rejection_reason(cleaned, text_signals(cleaned), removed) is not None


def test_subscription_link_does_not_reject_complete_article():
    text = ARTICLE + "\nSubscribers Only\nSubscribe for more news."
    assert page_rejection_reason(text, text_signals(text)) is None


@pytest.mark.parametrize("controls", ["Cookie Settings\nAccept All", "Privacy & Cookies Policy\nSAVE & ACCEPT"])
def test_cookie_dialog_cannot_supply_the_only_body_evidence(controls):
    text = (
        "Photo archive\n" + controls + "\n"
        "We use cookies to ensure that we give you the best experience on our website.\n"
        "These cookies do not store any personal information on your computer.\n"
    )
    cleaned, _ = clean_boilerplate(text)
    assert "We use cookies" not in cleaned
    assert not text_signals(cleaned).substantial_body
    assert not text_signals(cleaned).readable_short


def test_cookie_tutorial_is_preserved():
    text = (
        "Understanding Cookies in Web Applications\nCookie Settings\nWe use cookies to remember your settings.\n"
        + ARTICLE
    )
    cleaned, _ = clean_boilerplate(text)
    assert "We use cookies to remember your settings." in cleaned


def test_expired_job_cannot_be_rescued_by_related_job_descriptions():
    text = "Software engineering jobs\nSorry, but this job is no longer available. Browse other jobs.\n" + ARTICLE
    assert page_rejection_reason(text, text_signals(text)) == "unavailable_job"


def test_navigation_alone_does_not_reject_a_short_definition():
    text = "Legal definition\n" + "Reference heading\n" * 100 + SHORT
    assert page_rejection_reason(text, text_signals(text), removed_navigation_lines=30) is None


def test_explicit_archive_with_truncated_previews_is_rejected():
    preview = ("Here is an introduction to the topic that explains several important ideas "
               "and offers examples before the full article continues...\n")
    text = "Latest reports archive - Page 2\n" + preview * 2 + "Topic heading\n" * 40
    assert page_rejection_reason(text, text_signals(text)) == "teaser_index"


def test_recall_policy_reviews_model_and_rule_rejections_without_bypassing_nsfw(monkeypatch):
    monkeypatch.setattr(pipeline, "gopher_quality_filter", lambda text: False)
    monkeypatch.setattr(pipeline, "classify_quality", lambda text: ("cc", 0.99))
    monkeypatch.setattr(pipeline, "classify_nsfw", lambda text: ("non-nsfw", 0.99))
    monkeypatch.setattr(pipeline, "classify_toxic_speech", lambda text: ("non-toxic", 0.99))
    counts = Counter()
    assert pipeline.filter_text(ARTICLE, counts, "recall") is not None
    assert counts["gopher_rescued"] == counts["quality_rescued"] == 1
    assert pipeline.filter_text(ARTICLE, quality_policy="strict") is None
    monkeypatch.setattr(pipeline, "classify_nsfw", lambda text: ("nsfw", 0.99))
    assert pipeline.filter_text(ARTICLE, quality_policy="recall") is None


def test_readable_short_document_survives_post_dedup_review(tmp_path):
    path = tmp_path / "wet_0000.jsonl"
    write_shard(path, [SHORT, "Only a title", ""])
    stats = pipeline._filter_short_documents([path], tmp_path / "output", 50, preserve_readable=True)
    assert list(pipeline._read_documents(tmp_path / "output" / path.name)) == [SHORT]
    assert stats["readable_short_kept"] == 1
    assert stats["short_discarded"] == stats["empty_discarded"] == 1


def test_cleanup_does_not_newly_reject_prose_passing_original_gopher(monkeypatch):
    body = (
        "Catalog entry\nA babysitter solves a difficult problem by asking other people for help.\n" + "Metadata\n" * 60
    )
    monkeypatch.setattr(pipeline, "gopher_quality_filter", lambda text: text.startswith("Home\n"))
    monkeypatch.setattr(pipeline, "classify_quality", lambda text: ("wiki", 0.6))
    monkeypatch.setattr(pipeline, "classify_nsfw", lambda text: ("non-nsfw", 0.9))
    monkeypatch.setattr(pipeline, "classify_toxic_speech", lambda text: ("non-toxic", 0.9))
    counts = Counter()
    result = pipeline.filter_text("Home\n" + body, counts)
    assert result is not None and "A babysitter solves" in result
    assert counts["gopher_preclean_pass"] == 1


def test_conservative_line_dedup_preserves_prose_once_and_all_code_lines(tmp_path):
    prose = "The article explains why this method works and how people can apply it safely.\n"
    code = "def example():\n    return None\n    return None\n"
    path = tmp_path / "wet_0000.jsonl"
    write_shard(path, ["Home\n" + prose * 2 + code, "Home\n" + prose + "Unique final text.\n"])
    pipeline._exact_line_deduplication([path], tmp_path / "output", mode="conservative")
    first, second = pipeline._read_documents(tmp_path / "output" / path.name)
    assert first.count(prose) == second.count(prose) == 1
    assert code in first
    assert "Home" not in first and "Home" not in second


def test_strict_line_dedup_matches_course_function(tmp_path):
    from cs336_data.deduplication import exact_deduplication

    texts = ["shared\ninside\ninside\nfirst\n", "shared\nsecond\n"]
    paths = []
    for index, text in enumerate(texts):
        path = tmp_path / f"doc{index}.txt"
        path.write_text(text)
        paths.append(path)
    reference = tmp_path / "reference"
    reference.mkdir()
    exact_deduplication(paths, reference)
    shard = tmp_path / "wet_0000.jsonl"
    write_shard(shard, texts)
    pipeline._exact_line_deduplication([shard], tmp_path / "output", mode="strict")
    assert list(pipeline._read_documents(tmp_path / "output" / shard.name)) == [
        (reference / path.name).read_text() for path in paths
    ]
