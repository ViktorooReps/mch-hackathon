from newspaper import Article

from datamodel import Example


def convert_to_example(article: Article) -> Example:
    return Example(
        date=article.publish_date,
        text=article.text,
        title=article.title,
        authors=tuple(article.authors),
        url=article.url,
        source_url=article.source_url,
        tags=tuple(article.tags),
        keywords=article.meta_keywords,
        description=article.meta_description
    )