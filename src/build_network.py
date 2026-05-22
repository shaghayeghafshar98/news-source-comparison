import pandas as pd
from pathlib import Path
from pyvis.network import Network


ARTICLES_PATH = Path("data/processed/topic_articles_analyzed.csv")
PAIRS_PATH = Path("data/processed/similarity_pairs.csv")
OUTPUT_PATH = Path("outputs/networks/news_similarity_network.html")

SIMILARITY_THRESHOLD = 0.30


def shorten(text, max_len=70):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def make_key(source, title, link):
    return f"{source}|||{title}|||{link}"


def main():
    if not ARTICLES_PATH.exists():
        print(f"Missing file: {ARTICLES_PATH}")
        print("First run: python src/compare_articles.py")
        return

    if not PAIRS_PATH.exists():
        print(f"Missing file: {PAIRS_PATH}")
        print("First run: python src/compare_articles.py")
        return

    articles = pd.read_csv(ARTICLES_PATH)
    pairs = pd.read_csv(PAIRS_PATH)

    if len(articles) == 0:
        print("No articles found.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create article IDs
    articles = articles.reset_index(drop=True)
    articles["article_id"] = articles.index.map(lambda x: f"article_{x}")

    # Mapping from article information to node ID
    key_to_id = {}

    for _, row in articles.iterrows():
        key = make_key(row["source"], row["title"], row["link"])
        key_to_id[key] = row["article_id"]

    # Create network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="white",
        font_color="black",
        notebook=False,
        cdn_resources="in_line"
    )

    # Add nodes
    for _, row in articles.iterrows():
        title_html = (
            f"<b>Source:</b> {row['source']}<br>"
            f"<b>Title:</b> {row['title']}<br>"
            f"<b>Sentiment:</b> {row['sentiment_label']} "
            f"({row['sentiment_compound']:.2f})<br>"
            f"<b>Link:</b> {row['link']}"
        )

        source_short = {
            "BBC World": "BBC",
            "Guardian World": "Guardian",
            "NPR World": "NPR",
            "CNN World": "CNN",
            "Al Jazeera": "AJ",
        }.get(row["source"], row["source"])

        node_label = f"{source_short} {row.name + 1}"

        net.add_node(
            row["article_id"],
            label=node_label,
            title=title_html,
            group=row["source"],
            value=12
        )

    # Add edges for similar articles
    edge_count = 0

    for _, row in pairs.iterrows():
        similarity = float(row["similarity"])

        if similarity < SIMILARITY_THRESHOLD:
            continue

        key_1 = make_key(row["source_1"], row["title_1"], row["link_1"])
        key_2 = make_key(row["source_2"], row["title_2"], row["link_2"])

        if key_1 not in key_to_id or key_2 not in key_to_id:
            continue

        net.add_edge(
            key_to_id[key_1],
            key_to_id[key_2],
            value=similarity * 10,
            title=f"Similarity: {similarity:.3f}"
        )

        edge_count += 1

    net.show_buttons(filter_=["physics"])
    net.write_html(str(OUTPUT_PATH))

    print("======================")
    print("Network created")
    print("Articles/nodes:", len(articles))
    print("Similarity edges:", edge_count)
    print("Similarity threshold:", SIMILARITY_THRESHOLD)
    print("Saved to:", OUTPUT_PATH)
    print("======================")


if __name__ == "__main__":
    main()