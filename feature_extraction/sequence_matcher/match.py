from dateparser import parse
from thefuzz import fuzz


def soft_match(seq1: str, seq2: str) -> float:
    # Levenstein distance
    return fuzz.token_set_ratio(seq1, seq2, force_ascii=False) / 100


def hard_match(seq1: str, seq2: str) -> float:
    return 1.0 if seq1 == seq2 else 0.0


def date_match(seq1: str, seq2: str) -> float:
    return 1.0 if parse(seq1) == parse(seq2) else 0.0
