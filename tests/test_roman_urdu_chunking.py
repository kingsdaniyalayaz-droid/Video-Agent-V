import pytest
from core.roman_urdu_translator import split_transcript

def test_chunking_33320_chars():
    transcript = "This is a sentence. " * 1666
    assert 33300 <= len(transcript) <= 33400
    chunks = split_transcript(transcript, 3500)
    assert 8 <= len(chunks) <= 10

def test_chunking_bounds():
    transcript = "This is a sentence. " * 1666
    chunks = split_transcript(transcript, 3500)
    for chunk in chunks:
        assert len(chunk) <= 4500

def test_sentence_boundaries_preserved():
    transcript = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100 + "."
    chunks = split_transcript(transcript, max_chars=150)
    assert chunks[0] == "A" * 100 + ".\n" + "B" * 100 + "."

def test_large_paragraphs_split():
    transcript = "A" * 5000
    chunks = split_transcript(transcript, 3500)
    assert len(chunks) == 2
    assert len(chunks[0]) <= 3500

def test_no_empty_chunks():
    transcript = "   \n\n  \n"
    chunks = split_transcript(transcript, 3500)
    assert len(chunks) == 0
