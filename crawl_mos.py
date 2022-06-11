import argparse
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from crawlers import mos
from crawlers.adapter import convert_to_example
from crawlers.article_crawler import DATE_START, crawl_articles, DATE_END, UTC
from datamodel import JsonlDataset


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").astimezone(UTC)
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    parser = ArgumentParser('mos.ru crawler')
    parser.add_argument('--from_date', type=valid_date, default=DATE_START)
    parser.add_argument('--to_date', type=valid_date, default=DATE_END)
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--out_path', type=Path, default=Path('data/mos.jsonl'))

    args = parser.parse_args()

    urls_iterator = mos.get_urls()
    article_iterator = crawl_articles(urls_iterator,
                                      sorted_by_date=False,
                                      timeout=args.timeout,
                                      date_start=args.from_date,
                                      date_end=args.to_date)

    examples_iterator = map(convert_to_example, article_iterator)
    JsonlDataset(examples_iterator, args.out_path).write()
