import pytest

from b2w.textutil import chunk, cosine, load_book, ngrams, too_close

GUTENBERG = (
    "The Project Gutenberg eBook of A Test\n\nlicence text that must go\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK A TEST ***\n\n"
    "First paragraph has six words.\n\n"
    "Second paragraph also has six.\n\n"
    "Third paragraph is the one that should be cut off.\n\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK A TEST ***\n\nfooter that must go\n"
)


def test_load_book_strips_licence_and_cuts_at_paragraph_end(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(GUTENBERG, encoding="utf-8")
    text = load_book(path, max_words=12)
    assert "licence" not in text and "footer" not in text
    assert text == "First paragraph has six words.\n\nSecond paragraph also has six."


def test_load_book_keeps_everything_when_limit_is_large(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(GUTENBERG, encoding="utf-8")
    assert "Third paragraph" in load_book(path, max_words=1000)


def test_chunk_respects_the_word_target():
    paragraphs = ["word " * 100] * 5
    parts = chunk("\n\n".join(paragraphs), target_words=220)
    assert [len(p.split()) for p in parts] == [200, 200, 100]


def test_cosine_identical_and_disjoint():
    assert cosine("the red door", "the red door") == pytest.approx(1.0)
    assert cosine("red door", "blue window") == 0.0
    assert cosine("", "anything") == 0.0


def test_ngrams_are_four_token_windows():
    assert ngrams("a b c d e") == {("a", "b", "c", "d"), ("b", "c", "d", "e")}
    assert ngrams("a b c") == set()


def test_too_close_catches_a_copied_question():
    train = ["What colour was the door of the laboratory in the old house?"]
    assert too_close("What colour was the door of the laboratory in the old house?", train)


def test_too_close_catches_shared_wording():
    train = ["Who built the machine in the garden shed during the long winter?"]
    exam = "Who built the machine in the garden shed?"
    assert too_close(exam, train)


def test_too_close_lets_an_unrelated_question_through():
    train = ["Who built the machine in the garden shed during the long winter?"]
    assert not too_close("How many days did the narrator stay at the inn?", train)
