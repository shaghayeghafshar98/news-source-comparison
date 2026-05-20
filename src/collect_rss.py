import re
import html
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


# =========================
# Settings
# =========================

FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "NPR World": "https://feeds.npr.org/1004/rss.xml",
    "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

OUTPUT_PATH = Path("data/raw/all_rss_articles.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (student news comparison project)"
}


# =========================
# Helper functions
# =========================

def clean_html(text):
    if not text:
        return ""

    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def fetch_feed(source_name, feed_url):
    print(f"\nFetching: {source_name}")

    try:
        response = requests.get(feed_url, headers=HEADERS, timeout=20)
        print("Status:", response.status_code)

        if response.status_code != 200:
            print(f"Skipped {source_name}")
            return []

        feed = feedparser.parse(response.content)

        rows = []

        for entry in feed.entries:
            title = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary", ""))
            link = entry.get("link", "")
            published = entry.get("published", "")

            rows.append({
                "source": source_name,
                "title": title,
                "summary": summary,
                "link": link,
                "published": published,
                "text": f"{title}. {summary}"
            })

        print("Articles:", len(rows))
        return rows

    except Exception as e:
        print(f"Error for {source_name}: {e}")
        return []


# =========================
# Main program
# =========================

def main():
    all_rows = []

    for source_name, feed_url in FEEDS.items():
        all_rows.extend(fetch_feed(source_name, feed_url))

    df = pd.DataFrame(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n======================")
    print("Finished collection")
    print("Total articles:", len(df))
    print("Saved to:", OUTPUT_PATH)
    print("======================")

    if len(df) > 0:
        print("\nPreview:")
        print(df[["source", "title"]].head(10))


if __name__ == "__main__":
    main()