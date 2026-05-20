import argparse
import re
import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/raw/all_rss_articles.csv")
OUTPUT_PATH = Path("data/processed/topic_articles.csv")


def normalize_text(text):
    """Convert text to lowercase and remove extra spaces."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def topic_match(text, topic):
    """
    Simple topic matching.

    Example:
    topic = "Iran"
    -> article must contain "iran"

    topic = "Iran nuclear"
    -> article must contain both "iran" and "nuclear"
    """
    text = normalize_text(text)
    topic_words = normalize_text(topic).split()

    return all(word in text for word in topic_words)


def main():
    parser = argparse.ArgumentParser(description="Filter RSS articles by topic.")

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help='Topic to search for, for example: "Iran", "Gaza", "Ukraine", "climate change"'
    )

    args = parser.parse_args()
    topic = args.topic

    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("First run this command:")
        print("python src/collect_rss.py")
        return

    df = pd.read_csv(INPUT_PATH)

    if "text" not in df.columns:
        print("Error: The input CSV must contain a 'text' column.")
        return

    topic_df = df[df["text"].apply(lambda x: topic_match(x, topic))].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    topic_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("======================")
    print("Topic filtering finished")
    print("Topic:", topic)
    print("Total articles before filtering:", len(df))
    print("Matching articles:", len(topic_df))
    print("Saved to:", OUTPUT_PATH)
    print("======================")

    if len(topic_df) > 0:
        print("\nPreview:")
        print(topic_df[["source", "title"]].head(20).to_string(index=False))
    else:
        print("\nNo matching articles found.")
if __name__ == "__main__":
    main()