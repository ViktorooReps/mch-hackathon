import argparse
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from crawlers.adapter import convert_to_example
from crawlers.article_crawler import DATE_START, crawl_articles, DATE_END, UTC
from crawlers.mos.utils import get_urls
from datamodel import JsonlDataset


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").astimezone(UTC)
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    parser = ArgumentParser('mos.ru crawler')
    parser.add_argument('--ignore_date', action='store_true')
    parser.add_argument('--from_date', type=valid_date, default=DATE_START)
    parser.add_argument('--to_date', type=valid_date, default=DATE_END)
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--out_path', type=Path, default=Path('data/mos.jsonl'))

    args = parser.parse_args()

    urls_iterator = get_urls(timeout=args.timeout, patience=args.patience)
    article_iterator = crawl_articles(urls_iterator,
                                      ignore_date=args.ignore_date,
                                      sorted_by_date=False,
                                      date_start=args.from_date,
                                      date_end=args.to_date)

    examples_iterator = map(convert_to_example, article_iterator)
    JsonlDataset(examples_iterator, args.out_path).write()
