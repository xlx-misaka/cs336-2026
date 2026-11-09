import pytest

from cs336_data.mask_pii import mask_phone_numbers


@pytest.mark.parametrize(
    "number",
    [
        "01676 935000",
        "01676935000",
        "020 7946 0958",
        "07700 900123",
        "+44 1676 935000",
        "0044 1676 935000",
        "+44 (0)1676 935000",
        "+44 20 7946 0958",
        "(01676) 935000",
        "(0161) 496 0000",
        "(020) 7946 0958",
        "+1 (650) 253-0000",
        "650.253.0000",
        "555-1234",
        "(650) 253-0000 ext. 123",
        "01676 935000 x42",
        "0494 151 331",
        "(02) 1234 5678",
        "+61 494 151 331",
        "+40 364 101203",
        "0040 364 101203",
        "+81-3-4540-4348",
        "(876) 927-1660-9",
    ],
)
def test_mask_phone_regression_formats(number):
    assert mask_phone_numbers(f"Tel: {number}.") == ("Tel: |||PHONE_NUMBER|||.", 1)


@pytest.mark.parametrize(
    "text",
    [
        "doi:10.1007/s11356-009-0178-5",
        "https://doi.org/10.1007/s11356-009-0178-5",
        "DOI: 10.1234/650-253-0000",
        "10.1234/01676935000",
        '"10.1234/(650)253-0000"',
        "partABC555-1234XYZ",
        "123456789012345",
        "already masked: |||PHONE_NUMBER|||",
        "Tokyo,150-0001, Japan",
        "Postal code: 150-0001",
        "sites-host-mtl03-3002-v4.8.517-2026-04-16-e9d060dd",
        "Version: 517-2026",
        "555-1234",
        "https://example.com/item/650-253-0000",
        "",
    ],
)
def test_mask_phone_regression_preserves_non_phone_content(text):
    assert mask_phone_numbers(text) == (text, 0)


def test_mask_phone_regression_counts_only_real_replacements():
    text = "doi:10.1007/s11356-009-0178-5\nTel: 01676 935000; US: +1 (650) 253-0000."
    expected = "doi:10.1007/s11356-009-0178-5\nTel: |||PHONE_NUMBER|||; US: |||PHONE_NUMBER|||."
    assert mask_phone_numbers(text) == (expected, 2)
    assert mask_phone_numbers(expected) == (expected, 0)


def test_mask_phone_regression_preserves_newlines_and_indentation():
    text = "Heading\n    Tel: 01676 935000\n\nContact us."
    expected = "Heading\n    Tel: |||PHONE_NUMBER|||\n\nContact us."
    assert mask_phone_numbers(text) == (expected, 1)


def test_mask_phone_regression_context_and_replacement_count():
    text = "Tokyo,150-0001, Japan; Tel: 555-1234. Version: 517-2026."
    expected = "Tokyo,150-0001, Japan; Tel: |||PHONE_NUMBER|||. Version: 517-2026."
    assert mask_phone_numbers(text) == (expected, 1)


def test_mask_phone_regression_international_without_cue():
    assert mask_phone_numbers("Contact\n+40 364 101203\nTokyo,150-0001, Japan") == (
        "Contact\n|||PHONE_NUMBER|||\nTokyo,150-0001, Japan",
        1,
    )
