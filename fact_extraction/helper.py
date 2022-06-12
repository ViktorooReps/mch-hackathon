from functools import partial
from typing import Iterable, Tuple, Optional

import numpy as np

from fact_extraction.model import Entity
from feature_extraction.sequence_matcher.levenstein import match

months = [
    "май", "мая", "маю", "маем", "мае",
    "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр", "январ", "феврал", "март", "апрел",
]

time_words = ['до', 'и', 'начнется']


def check_if_month(string: str) -> bool:
    for month in months:
        if month in string:
            return True
    return False


def check_if_time_words(string: str) -> bool:
    for word in time_words:
        if word in string:
            return True
    return False


def get_filtered_date(inputs: str) -> Tuple[bool, Optional[str]]:
    inputs = inputs.replace('-', ' - ')
    inputs = inputs.replace('\n', ' ')
    if '№' in inputs:
        return False, None
    tokens = inputs.lower().split()
    min_len = 1
    res = []
    for token in tokens:
        if token == '-':
            min_len += 1
        if sum([letter.isalpha() for letter in token]):
            if check_if_month(token):
                res.append(token)
        else:
            res.append(token)

    if len(res) < min_len:
        return False, None
    
    return True, ' '.join(res).strip('-').strip()


def get_filtered_time(inputs: str) -> Tuple[bool, Optional[str]]:
    inputs = inputs.replace('-', ' ')
    inputs = inputs.replace('\n', ' ')
    
    words = inputs.split()
    res = []
    for word in words:
        if word.isalpha():
            if word not in time_words:
                return False, None
            res.append(word)
        else:
            for letter in word:
                if not (letter in ',:. ' or letter.isdigit()): 
                    return False, None
            res.append(word)
        
    return True, ' '.join(res).strip('-').strip()


def get_filtered_cardinal(inputs: str) -> Tuple[bool, Optional[str]]:
    inputs = inputs.replace('\n', ' ').strip()
    for letter in inputs:
        if not (letter in ',.- ' or letter.isdigit()):
            return False, None
    
    return True, str(np.random.randint(0, 100_000, size=1)[0])


def get_fact_consistency(true_facts: Iterable[Entity], target_facts: Iterable[Entity]) -> float:

    def facts_as_strings(facts: Iterable[Entity]) -> Iterable[str]:
        for fact in facts:
            yield fact.text

    true_strs = set(facts_as_strings(true_facts))
    target_strs = set(facts_as_strings(target_facts))

    if not len(true_strs):
        return 1.0

    res = []
    for target_fact in target_strs:
        matcher = partial(match, seq2=target_fact)
        fact_consistency = max(map(matcher, true_strs))
        res.append(fact_consistency)

    return sum(res) / len(res)
