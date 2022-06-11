from time import sleep
from typing import Iterable, Dict, Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


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


CHROME_PATH = '/usr/bin/google-chrome'
CHROMEDRIVER_PATH = '/usr/bin/chromedriver'
WINDOW_SIZE = "1920,1080"


def get_urls(*, timeout: float = 1.0) -> Iterable[str]:
    request_builder = RequestBuilder('https://www.mos.ru/search?')
    args = {
        'category': 'newsfeed',
        'skip_stat': 2,
        'spheres': 14299,
        'types': 'news'
    }

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    driver = webdriver.Chrome(options=chrome_options)

    page = 1
    while True:
        print(f'Crawling page {page}...')
        args['page'] = page

        url = request_builder.build(args)
        driver.get(url)
        text = driver.page_source
        try:
            yield from _get_urls_from_html(text)
        except ValueError:
            return  # no page found

        sleep(timeout)
        page += 1


if __name__ == '__main__':
    for url in get_urls():
        pass
