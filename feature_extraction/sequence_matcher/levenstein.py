from thefuzz import fuzz


def match(seq1: str, seq2: str) -> float:
    # Levenstein distance
    return fuzz.token_set_ratio(seq1, seq2, force_ascii=False) / 100
