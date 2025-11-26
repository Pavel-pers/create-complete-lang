import requests
from bs4 import BeautifulSoup

from cclang.common.logx import get_logger

"""
parse list of links avalibale books from website
https://epustakalay.com/list-of-all-marathi-books
html-code sample of books list inside create-complete-lang/data/samples_data/epustakalay_books_list.html
html-code sample of book website page inside /Users/ppers/scientific_work/create-complete-lang/data/samples_data/epustakalay_book_page.html
"""

log = get_logger("parser.epustakalay").bind(component="parse_book_list")

def parse_list_of_page(page: int) -> list[str]:
    """
    parse list of links avalibale books from website
    https://epustakalay.com/list-of-all-marathi-books/page/<PAGE>/
    where <PAGE> is the number of list page
    :param page:
    :return: list of links to pages of books
    """
    if page < 1:
        raise ValueError("Page number must be greater than zero")

    url = f"https://epustakalay.com/list-of-all-marathi-books/page/{page}/"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    book_card_selector = "div.col-6.col-sm-4.col-md-3.col-lg-2.p-2 a"

    links: list[str] = []
    for anchor in soup.select(book_card_selector):
        href = anchor.get("href")
        if not href:
            continue
        links.append(href.strip())

    return links

def parse_book_link(book_link: str) -> str:
    """
    parse book pdf file link from book page
    :param book_link:
    :return: link to book pdf file
    """
    response = requests.get(book_link, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    download_div_selector = "div.col-12.col-sm-6.mb-1.p-1"

    for container in soup.select(download_div_selector):
        anchor = container.find("a", href=True)
        if not anchor:
            continue

        href = anchor["href"].strip()
        text = anchor.get_text(strip=True).lower()

        if href.lower().endswith(".pdf"):
            return href
        if "archive.org/download" in href:
            return href
        if "download" in text or "pdf" in text:
            return href

    raise ValueError(f"Could not find download link on page: {book_link}")

def parse_book_list(page_from: int, page_to: int, limit: int | None) -> list[str]:
    """
    parse all links of current segment of pages
    :param page_from:
    :param page_to:
    :return: a list of links to book pdfs
    """
    files_links = []
    for page in range(page_from, page_to + 1):
        book_links = parse_list_of_page(page)
        for book_link in book_links:
            files_links.append(parse_book_link(book_link))
            total_links = len(files_links)
            if total_links % 10 == 0:
                log.info(f"parsed link count checkpoint, total : {str(total_links)}")
            if limit is not None and total_links >= limit:
                log.info(f"parsed link count final, total : {str(total_links)}")
                return files_links

    if files_links:
        log.info(f"parsed link count final, total : {str(len(files_links))}")

    return files_links
