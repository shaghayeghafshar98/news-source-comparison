import html
import pandas as pd
from pathlib import Path


ARTICLES_PATH = Path("data/processed/topic_articles_analyzed.csv")
PAIRS_PATH = Path("data/processed/similarity_pairs.csv")
SOURCE_SUMMARY_PATH = Path("data/processed/source_summary.csv")

REPORT_OUTPUT = Path("outputs/reports/news_comparison_report.html")
NETWORK_PATH = Path("../networks/news_similarity_network.html")

TOPIC_NAME = "Iran"


def source_short_name(source):
    return {
        "BBC World": "BBC",
        "Guardian World": "Guardian",
        "NPR World": "NPR",
        "CNN World": "CNN",
        "Al Jazeera": "AJ",
    }.get(source, source)


def sentiment_explanation():
    return """
    <p>
    The sentiment score is calculated from the article title and RSS summary.
    It ranges roughly from negative to positive. This is only a simple automatic
    estimate and should not be interpreted as a final political judgment.
    </p>
    """


def make_clickable_link(url):
    if pd.isna(url) or str(url).strip() == "":
        return ""
    safe_url = html.escape(str(url), quote=True)
    return f'<a href="{safe_url}" target="_blank">Open article</a>'


def main():
    if not ARTICLES_PATH.exists():
        print(f"Missing file: {ARTICLES_PATH}")
        print("First run: python src/compare_articles.py")
        return

    if not PAIRS_PATH.exists():
        print(f"Missing file: {PAIRS_PATH}")
        print("First run: python src/compare_articles.py")
        return

    if not SOURCE_SUMMARY_PATH.exists():
        print(f"Missing file: {SOURCE_SUMMARY_PATH}")
        print("First run: python src/compare_articles.py")
        return

    articles = pd.read_csv(ARTICLES_PATH).reset_index(drop=True)
    pairs = pd.read_csv(PAIRS_PATH)
    source_summary = pd.read_csv(SOURCE_SUMMARY_PATH)

    if len(articles) == 0:
        print("No analyzed articles found.")
        return

    # Create node labels matching the network
    articles["node_label"] = articles.apply(
        lambda row: f"{source_short_name(row['source'])} {row.name + 1}",
        axis=1
    )

    # Prepare article table
    article_table = articles[
        [
            "node_label",
            "source",
            "title",
            "published",
            "sentiment_label",
            "sentiment_compound",
            "link",
        ]
    ].copy()

    article_table["link"] = article_table["link"].apply(make_clickable_link)
    article_table["sentiment_compound"] = article_table["sentiment_compound"].round(3)

    # Prepare source summary
    source_summary_display = source_summary.copy()
    source_summary_display["average_sentiment"] = source_summary_display["average_sentiment"].round(3)

    # Prepare similarity table
    pairs_display = pairs.head(15)[
        [
            "source_1",
            "title_1",
            "source_2",
            "title_2",
            "similarity",
        ]
    ].copy()

    pairs_display["similarity"] = pairs_display["similarity"].round(3)

    report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>News Source Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.5;
            color: #222;
        }}

        h1, h2 {{
            color: #1f2937;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 35px;
            font-size: 14px;
        }}

        th, td {{
            border: 1px solid #ccc;
            padding: 8px;
            vertical-align: top;
        }}

        th {{
            background-color: #f2f2f2;
        }}

        .box {{
            background-color: #f8f9fa;
            border-left: 4px solid #555;
            padding: 12px;
            margin-bottom: 25px;
        }}

        a {{
            color: #0066cc;
        }}
    </style>
</head>

<body>

<h1>News Source Comparison Report</h1>

<div class="box">
    <p><b>Topic:</b> {TOPIC_NAME}</p>
    <p><b>Total matching articles:</b> {len(articles)}</p>
    <p><b>Sources included:</b> {", ".join(sorted(articles["source"].unique()))}</p>
    <p>
        <a href="{NETWORK_PATH}" target="_blank">Open similarity network visualization</a>
    </p>
</div>

<h2>1. What this project does</h2>

<p>
This project collects recent news items from RSS feeds, filters them by a selected topic,
and compares how different news sources cover that topic.
</p>

<p>
In the network visualization, each node represents one article. The node color represents
the news source. Edges connect articles that are textually similar based on their title
and RSS summary. Thicker edges indicate stronger similarity.
</p>

<h2>2. Source Summary</h2>

<p>
This table shows how many articles each source had about the selected topic and the
average automatic sentiment score.
</p>

{source_summary_display.to_html(index=False, escape=False)}

<h2>3. Article Node Map</h2>

<p>
This table explains the node labels used in the network visualization.
</p>

{article_table.to_html(index=False, escape=False)}

<h2>4. Most Similar Article Pairs</h2>

<p>
This table shows the article pairs with the highest textual similarity.
</p>

{pairs_display.to_html(index=False, escape=False)}

<h2>5. Sentiment Note</h2>

{sentiment_explanation()}

</body>
</html>
"""

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(report_html)

    print("======================")
    print("Report created")
    print("Saved to:", REPORT_OUTPUT)
    print("======================")


if __name__ == "__main__":
    main()