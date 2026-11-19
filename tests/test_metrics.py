import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import normalize, edit_ops, wer, cer, matches_any


def test_same_text_is_zero():
    assert wer("the cat sat", "the cat sat") == 0
    assert cer("the cat sat", "the cat sat") == 0


def test_case_and_punctuation_ignored():
    assert wer("Hello, World!", "hello world") == 0


def test_one_substitution():
    assert wer("the cat sat on the mat", "the dog sat on the mat") == pytest.approx(1 / 6)


def test_deletion_and_insertion():
    assert edit_ops("a b c".split(), "a c".split()) == (0, 1, 0)
    assert edit_ops("a c".split(), "a b c".split()) == (0, 0, 1)


def test_wer_can_go_above_one():
    # 1 ref word, 3 hyp words -> 1 sub + 2 ins = 3 errors
    assert wer("yes", "no no no") == 3


def test_empty_hypothesis():
    assert wer("one two three", "") == 1


def test_cer():
    # "kat" vs "cat" -> 1 char wrong out of 3
    assert cer("cat", "kat") == pytest.approx(1 / 3)


def test_normalize():
    assert normalize("Twenty-Three  Rupees.") == "twenty three rupees"
    assert normalize("I'm here, don't stop") == "i am here do not stop"
    assert normalize("the boys' ball") == "the boys ball"


def test_accept_matches_digits_or_words():
    accept = "two lakh fifty thousand|250000"
    assert matches_any("Your loan of ₹2,50,000 is approved", accept)
    assert matches_any("your loan of two lakh fifty thousand rupees", accept)
    assert not matches_any("your loan of twenty five thousand", accept)


def test_accept_needs_every_group():
    accept = "345000;31st march|march 31"
    assert matches_any("pay 3,45,000 before 31st March", accept)
    assert not matches_any("pay 3,45,000 before the end of the month", accept)
    # decimal point is kept, so 1.2 crore is not confused with 12 crore
    assert not matches_any("12 crore", "1.2 crore")


def test_matches_jiwer():
    jiwer = pytest.importorskip("jiwer")
    pairs = [
        ("he did not play in the world series", "he did play in world series again"),
        ("please bring soap", "please ring the soap"),
    ]
    for ref, hyp in pairs:
        assert wer(ref, hyp) == pytest.approx(jiwer.wer(ref, hyp))
