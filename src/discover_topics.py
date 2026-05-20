import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


INPUT_PATH = Path("data/raw/all_rss_articles.csv")
OUTPUT_PATH = Path("data/processed/discovered_topics.csv")


def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("First run: python src/collect_rss.py")
        return

    df = pd.read_csv(INPUT_PATH)

    if "text" not in df.columns:
        print("The input CSV must contain a 'text' column.")
        return

    texts = df["text"].fillna("").astype(str).tolist()

    # Frequent words and phrases
    count_vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),      # single words + two-word phrases
        min_df=2,                # must appear in at least 2 articles
        max_features=100
    )

    count_matrix = count_vectorizer.fit_transform(texts)
    count_scores = count_matrix.sum(axis=0).A1
    count_terms = count_vectorizer.get_feature_names_out()

    count_results = pd.DataFrame({
        "term": count_terms,
        "frequency_score": count_scores
    }).sort_values("frequency_score", ascending=False)

    # TF-IDF: terms that are important, not only frequent
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()

    tfidf_results = pd.DataFrame({
        "term": tfidf_terms,
        "tfidf_score": tfidf_scores
    }).sort_values("tfidf_score", ascending=False)

    # Merge both results
    results = pd.merge(
        count_results,
        tfidf_results,
        on="term",
        how="outer"
    ).fillna(0)

    results = results.sort_values(
        ["tfidf_score", "frequency_score"],
        ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("======================")
    print("Topic discovery finished")
    print("Input articles:", len(df))
    print("Saved to:", OUTPUT_PATH)
    print("======================")

    print("\nTop suggested topic terms:")
    print(results.head(30).to_string(index=False))


if __name__ == "__main__":
    main()