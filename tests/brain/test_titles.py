from argos.brain.titles import derive_title


def test_derive_title_takes_first_non_empty_line():
    assert derive_title("\n\n  Anthropic ships Claude 5  \nbody text\n") == "Anthropic ships Claude 5"


def test_derive_title_truncates_to_500_chars():
    long_line = "x" * 600
    assert derive_title(long_line) == "x" * 500


def test_derive_title_falls_back_to_untitled_when_blank():
    assert derive_title("   \n\t\n") == "Untitled"
    assert derive_title("") == "Untitled"
    assert derive_title(None) == "Untitled"
