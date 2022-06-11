from requests_html import HTMLSession

from time import sleep
from typing import Iterable, Dict, Any

from requests import get, post
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import logging
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import colorama


class RequestBuilder:

    def __init__(self, url: str):
        self._url = url

    def build(self, arg_values: Dict[str, Any]):
        parsed_args = []
        for arg_name, arg_value in arg_values.items():
            parsed_args.append(f'{arg_name}={arg_value}')

        parsed_args = '&'.join(parsed_args)
        return self._url + parsed_args

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def _get_urls_from_html(url, html: str) -> Iterable[str]:
    if not is_valid(url):
        print("END")
        return
    urls = set()
    soup = BeautifulSoup(html, "html.parser")
    #print(requests.get(url).content)
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

    return urls

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

    for page in range(41, 47):
        print(f'Crawling page {page}...')
        args['page'] = page

        url = request_builder.build(args)
        driver.get(url)
        text = driver.page_source
        yield from _get_urls_from_html(url, text)

        sleep(timeout)


if __name__ == '__main__':
    for url in get_urls():
        print(url)
