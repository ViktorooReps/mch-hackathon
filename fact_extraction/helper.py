from collections import defaultdict
from functools import partial
from typing import Iterable, Tuple, Optional, Dict, Set

import numpy as np

from fact_extraction.model import Entity, get_matcher, EntityType

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
    true_fact_mapping: Dict[EntityType, Set[str]] = defaultdict(set)
    target_fact_mapping: Dict[EntityType, Set[str]] = defaultdict(set)

    for fact in true_facts:
        true_fact_mapping[fact.label].add(fact.text)

    for fact in target_facts:
        target_fact_mapping[fact.label].add(fact.text)

    if not len(true_fact_mapping):
        return 1.0

    res = []
    for target_label, target_label_facts in target_fact_mapping.items():
        for target_fact in target_label_facts:
            if not len(true_fact_mapping[target_label]):
                return 1.0

            matcher = partial(get_matcher(target_label), seq2=target_fact)
            fact_consistency = max(map(matcher, true_fact_mapping[target_label]))
            res.append(fact_consistency)

    if not len(res):
        return 1.0

    return sum(res) / len(res)
