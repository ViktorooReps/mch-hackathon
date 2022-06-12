from enum import Enum
from typing import NamedTuple, Callable

from feature_extraction.sequence_matcher.match import hard_match, date_match, soft_match


class EntityType(Enum):
    TIME = 'TIME'
    DATE = 'DATE'
    PERCENT = 'PERCENT'
    CARDINAL = 'CARDINAL'
    LOCATION = 'LOC'
    ORGANISATION = 'ORG'
    PERSON = 'PER'


HARD_MATCH_TYPES = {EntityType.PERCENT, EntityType.CARDINAL}
DATE_MATCH_TYPES = {EntityType.DATE, EntityType.TIME}


def get_matcher(type_: EntityType) -> Callable[[str, str], float]:
    if type_ in HARD_MATCH_TYPES:
        return hard_match
    if type_ in DATE_MATCH_TYPES:
        return date_match
    return soft_match


class Entity(NamedTuple):
    text: str
    start: int
    end: int
    label: EntityType
