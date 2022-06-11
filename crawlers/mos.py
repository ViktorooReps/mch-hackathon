from time import sleep
from typing import Iterable, Dict, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


class RequestBuilder:

    def __init__(self, url: str):
        self._url = url

    def build(self, arg_values: Dict[str, Any]):
        parsed_args = []
        for arg_name, arg_value in arg_values.items():
            parsed_args.append(f'{arg_name}={arg_value}')

        parsed_args = '&'.join(parsed_args)
        return self._url + parsed_args


def is_valid(url: str):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def _get_urls_from_html(url: str, html: str) -> Iterable[str]:
    urls = set()
    soup = BeautifulSoup(html, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue
        if 'news' not in href or 'item' not in href:
            continue
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if is_valid(href):
            urls.add(href)

    if not len(urls):
        raise ValueError(f'No URLs found at {url}')

    return urls


CHROME_PATH = '/usr/bin/google-chrome'
CHROMEDRIVER_PATH = '/usr/bin/chromedriver'
WINDOW_SIZE = "1920,1080"


def get_urls(*, timeout: float = 0.0, patience: int = 10) -> Iterable[str]:
    request_builder = RequestBuilder('https://www.mos.ru/search?')
    args = {
        'category': 'newsfeed',
        'date_from': '2012-01-01+12%3A00%3A00',
        'date_from_limit': '1442869200',
        'date_to': '2022-06-01+12%3A00%3A00',
        'date_to_limit': '1654981199',
        'hostApplied': 'false',
        'sort': 'date_desc',
        'spheres': '183299%2C14299',
        'types': 'news'
    }

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    driver = webdriver.Chrome(options=chrome_options)

    page = 1
    curr_patience = patience
    while True:
        print(f'Crawling page {page}...')
        args['page'] = page

        url = request_builder.build(args)
        driver.get(url)
        text = driver.page_source
        try:
            yield from _get_urls_from_html(url, text)
        except ValueError:
            if not curr_patience:
                return
            curr_patience -= 1
            continue

        sleep(timeout)
        page += 1
        curr_patience = patience


if __name__ == '__main__':
    for u in get_urls():
        print(u)
