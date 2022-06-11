import datetime
from typing import Iterable

import pytz as pytz
from newspaper import Article


UTC = pytz.UTC

DATE_START = datetime.datetime(year=2021, month=1, day=1, tzinfo=UTC)
DATE_END = datetime.datetime(year=2022, month=6, day=11, tzinfo=UTC)


def crawl_articles(
        article_urls: Iterable[str],
        *,
        sorted_by_date: bool = True,
        date_start: datetime.datetime = DATE_START,
        date_end: datetime.datetime = DATE_END
) -> Iterable[Article]:

    for article in map(Article, article_urls):
        article.download()
        article.parse()

        if article.publish_date < date_start:
            if sorted_by_date:
                print('Date limit exceeded!')
                break
            continue

        if article.publish_date <= date_end:
            yield article


if __name__ == '__main__':
    urls = [
        'https://www.rbc.ru/technology_and_media/11/06/2022/62a4573c9a79474e08ab1cff',
        'https://www.rbc.ru/technology_and_media/11/06/2022/62a4573c9a79474e08ab1cff',
        'https://www.rbc.ru/technology_and_media/11/06/2022/62a3a2269a794724bde4a0e8',
        'https://www.rbc.ru/technology_and_media/10/06/2022/62a3949c9a7947224daddf68',
        'https://www.rbc.ru/technology_and_media/10/06/2022/62a371bf9a794717cb0e746b',
        'https://www.rbc.ru/technology_and_media/10/06/2022/62a34b5d9a794708a93b11a6',
        'https://www.rbc.ru/technology_and_media/10/06/2022/62a2d49a9a7947d0e8291af9',
        'https://www.rbc.ru/technology_and_media/09/06/2022/62a21b6d9a7947a092d2c7c5',
        'https://www.rbc.ru/technology_and_media/09/06/2022/62a1eac49a79478250299d07',
        'https://www.rbc.ru/technology_and_media/09/06/2022/62a0efd99a7947305e31c91a',
        'https://www.rbc.ru/technology_and_media/09/06/2022/62a0cf759a794724a893180c',
        'https://www.rbc.ru/technology_and_media/09/06/2022/62a0bb419a79471aefb3cc5e',
        'https://www.rbc.ru/technology_and_media/08/06/2022/62a0c2799a79471eca4dc0cc',
        'https://www.rbc.ru/technology_and_media/08/06/2022/62a0bdea9a79471be7248477',
        'https://www.rbc.ru/technology_and_media/08/06/2022/629f7a099a7947322474d15d',
    ]

    for art in crawl_articles(urls):
        print(f'{art.publish_date}: {art.title}')
