import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


INPUT_PATH = Path("data/processed/topic_articles.csv")

ANALYZED_OUTPUT = Path("data/processed/topic_articles_analyzed.csv")
SIMILARITY_OUTPUT = Path("data/processed/similarity_pairs.csv")
SOURCE_SUMMARY_OUTPUT = Path("data/processed/source_summary.csv")


def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("First run: python src/filter_topic.py --topic Iran")
        return

    df = pd.read_csv(INPUT_PATH)

    if len(df) == 0:
        print("No topic articles found in topic_articles.csv.")
        print("Run again with a topic that has results, for example:")
        print("python src/filter_topic.py --topic Iran")
        return

    df["text"] = df["text"].fillna("").astype(str)

    # =========================
    # 1. Sentiment analysis
    # =========================

    analyzer = SentimentIntensityAnalyzer()

    df["sentiment_compound"] = df["text"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    df["sentiment_label"] = df["sentiment_compound"].apply(sentiment_label)

    # =========================
    # 2. Text similarity
    # =========================

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    pairs = []

    df_reset = df.reset_index(drop=True)

    for i in range(len(df_reset)):
        for j in range(i + 1, len(df_reset)):
            pairs.append({
                "source_1": df_reset.loc[i, "source"],
                "title_1": df_reset.loc[i, "title"],
                "link_1": df_reset.loc[i, "link"],
                "source_2": df_reset.loc[j, "source"],
                "title_2": df_reset.loc[j, "title"],
                "link_2": df_reset.loc[j, "link"],
                "similarity": similarity_matrix[i, j],
            })

    pairs_df = pd.DataFrame(pairs)

    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("similarity", ascending=False)

    # =========================
    # 3. Summary by source
    # =========================

    source_summary = (
        df.groupby("source")
        .agg(
            article_count=("title", "count"),
            average_sentiment=("sentiment_compound", "mean")
        )
        .reset_index()
        .sort_values("article_count", ascending=False)
    )

    # =========================
    # 4. Save outputs
    # =========================

    ANALYZED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ANALYZED_OUTPUT, index=False, encoding="utf-8")
    pairs_df.to_csv(SIMILARITY_OUTPUT, index=False, encoding="utf-8")
    source_summary.to_csv(SOURCE_SUMMARY_OUTPUT, index=False, encoding="utf-8")

    print("======================")
    print("Article comparison finished")
    print("Input articles:", len(df))
    print("Saved analyzed articles to:", ANALYZED_OUTPUT)
    print("Saved similarity pairs to:", SIMILARITY_OUTPUT)
    print("Saved source summary to:", SOURCE_SUMMARY_OUTPUT)
    print("======================")

    print("\nSource summary:")
    print(source_summary.to_string(index=False))

    print("\nMost similar article pairs:")
    if len(pairs_df) > 0:
        print(
            pairs_df[
                ["source_1", "title_1", "source_2", "title_2", "similarity"]
            ].head(10).to_string(index=False)
        )
    else:
        print("Not enough articles to compare.")


if __name__ == "__main__":
    main()