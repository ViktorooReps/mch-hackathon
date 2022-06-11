import datetime
from pathlib import Path
from typing import Optional, Iterable, List, Tuple

from pydantic import BaseModel


class Example(BaseModel):
    date: datetime.datetime
    text: str


class JsonlDataset:

    def __init__(self, examples: Iterable[Example], dataset_file: Optional[Path] = None):
        self._dataset_file = dataset_file
        self._examples = tuple(examples)

    @staticmethod
    def read(dataset_file: Path) -> 'JsonlDataset':
        examples: List[Example] = []
        with open(dataset_file) as f:
            for line in f:
                examples.append(Example.parse_raw(line))

        return JsonlDataset(examples, dataset_file)

    def write(self, to_file: Optional[Path] = None):
        if to_file is None:
            to_file = self._dataset_file

        if to_file is None:
            raise ValueError('Cannot determine save location!')

        with open(to_file, 'w') as f:
            for example in self._examples:
                f.write(example.json(exclude_none=True) + '\n')

    @property
    def examples(self) -> Tuple[Example]:
        return self._examples

    def __iter__(self):
        return self._examples

    def __len__(self):
        return len(self._examples)


if __name__ == '__main__':
    exs = [
        Example(date=datetime.datetime(year=2020, month=12, day=12), text='smth'),
        Example(date=datetime.datetime(year=2020, month=12, day=12), text='smth2'),
        Example(date=datetime.datetime(year=2020, month=12, day=12), text='smth3')
    ]

    dataset_file = Path('test.jsonl')
    JsonlDataset(exs).write(dataset_file)
    assert JsonlDataset.read(dataset_file).examples == tuple(exs)