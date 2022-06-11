from time import sleep
from typing import Iterable, Dict, Any

from requests import get, post
from selenium import webdriver


class RequestBuilder:

    def __init__(self, url: str):
        self._url = url

    def build(self, arg_values: Dict[str, Any]):
        parsed_args = []
        for arg_name, arg_value in arg_values.items():
            parsed_args.append(f'{arg_name}={arg_value}')

        parsed_args = '&'.join(parsed_args)
        return self._url + parsed_args


def _get_urls_from_html(html: str) -> Iterable[str]:
    yield 1


def get_urls(*, timeout: float = 1.0) -> Iterable[str]:
    request_builder = RequestBuilder('https://www.mos.ru/search?')
    args = {
        'category': 'newsfeed',
        'skip_stat': 2,
        'spheres': 14299,
        'types': 'news'
    }
    driver = webdriver.Chrome()
    for page in range(41, 47):
        print(f'Crawling page {page}...')
        args['page'] = page

        url = request_builder.build(args)
        driver.get(url)
        text = driver.page_source
        yield from _get_urls_from_html(text)

        sleep(timeout)


if __name__ == '__main__':
    for page in get_urls():
        pass
