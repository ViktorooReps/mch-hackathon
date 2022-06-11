from time import sleep
from typing import Iterable, Dict, Any

from newspaper import Article
from requests import get


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
    pass


def get_urls(*, timeout: float = 1.0) -> Iterable[str]:
    request_builder = RequestBuilder('https://www.mos.ru/search?')
    args = {
        'category': 'newsfeed',
        'q': '',
        'skip_stat': 2,
        'spheres': 14299,
        'types': 'news'
    }
    for page in range(1, 44):
        print(f'Crawling page {page}...')
        args['page'] = page

        url = request_builder.build(args)
        response = get(url)
        yield from _get_urls_from_html(response.text)

        sleep(timeout)




if __name__ == '__main__':
    get_urls()
