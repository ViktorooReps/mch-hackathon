from enum import Enum
from typing import NamedTuple


class EntityType(Enum):
    TIME = 'TIME'
    DATE = 'DATE'
    PERCENT = 'PERCENT'
    CARDINAL = 'CARDINAL'
    LOCATION = 'LOC'
    ORGANISATION = 'ORG'
    PERSON = 'PER'


class Entity(NamedTuple):
    text: str
    start: int
    end: int
    label: EntityType
