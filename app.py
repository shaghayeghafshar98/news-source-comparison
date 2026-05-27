import re
import html as html_lib
import time
from pathlib import Path

import requests
import feedparser
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyvis.network import Network
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
RAW_CSV = Path("data/raw/all_rss_articles.csv")
TOPICS_CSV = Path("data/processed/discovered_topics.csv")
FILTERED_CSV = Path("data/processed/topic_articles.csv")
ANALYZED_CSV = Path("data/processed/topic_articles_analyzed.csv")
PAIRS_CSV = Path("data/processed/similarity_pairs.csv")
SOURCE_SUMMARY_CSV = Path("data/processed/source_summary.csv")
NETWORK_HTML = Path("outputs/networks/news_similarity_network.html")

SIMILARITY_THRESHOLD = 0.30

FEEDS = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "NPR World": "https://feeds.npr.org/1004/rss.xml",
    "CNN World": "http://rss.cnn.com/rss/edition_world.rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (student news comparison project)"}

SOURCE_COLORS = {
    "BBC World": "#bb1919",
    "Guardian World": "#0084c6",
    "NPR World": "#1d6b2e",
    "CNN World": "#cc0001",
    "Al Jazeera": "#f7a600",
}


# ─────────────────────────────────────────────
# Pipeline functions
# ─────────────────────────────────────────────

def clean_html(text):
    if not text:
        return ""
    text = html_lib.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def collect_rss(progress_callback=None):
    all_rows = []
    for i, (source_name, feed_url) in enumerate(FEEDS.items()):
        if progress_callback:
            progress_callback(i, source_name)
        try:
            response = requests.get(feed_url, headers=HEADERS, timeout=20)
            if response.status_code != 200:
                continue
            feed = feedparser.parse(response.content)
            for entry in feed.entries:
                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", ""))
                all_rows.append({
                    "source": source_name,
                    "title": title,
                    "summary": summary,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "text": f"{title}. {summary}",
                })
        except Exception:
            continue
    df = pd.DataFrame(all_rows)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    return df


def discover_topics(df):
    texts = df["text"].fillna("").astype(str).tolist()
    count_vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=100)
    count_matrix = count_vec.fit_transform(texts)
    count_scores = count_matrix.sum(axis=0).A1
    count_results = pd.DataFrame({"term": count_vec.get_feature_names_out(), "frequency_score": count_scores})

    tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=100)
    tfidf_matrix = tfidf_vec.fit_transform(texts)
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_results = pd.DataFrame({"term": tfidf_vec.get_feature_names_out(), "tfidf_score": tfidf_scores})

    results = pd.merge(count_results, tfidf_results, on="term", how="outer").fillna(0)
    results = results.sort_values(["tfidf_score", "frequency_score"], ascending=False)
    TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TOPICS_CSV, index=False, encoding="utf-8")
    return results


def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def filter_topic(df, topic):
    topic_words = normalize_text(topic).split()
    mask = df["text"].apply(lambda x: all(w in normalize_text(x) for w in topic_words))
    filtered = df[mask].copy()
    FILTERED_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(FILTERED_CSV, index=False, encoding="utf-8")
    return filtered


def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def analyze(df):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment_compound"].apply(sentiment_label)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    sim_matrix = cosine_similarity(tfidf_matrix)

    df_reset = df.reset_index(drop=True)
    pairs = []
    for i in range(len(df_reset)):
        for j in range(i + 1, len(df_reset)):
            pairs.append({
                "source_1": df_reset.loc[i, "source"],
                "title_1": df_reset.loc[i, "title"],
                "link_1": df_reset.loc[i, "link"],
                "source_2": df_reset.loc[j, "source"],
                "title_2": df_reset.loc[j, "title"],
                "link_2": df_reset.loc[j, "link"],
                "similarity": sim_matrix[i, j],
            })

    pairs_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False) if pairs else pd.DataFrame()
    source_summary = (
        df.groupby("source")
        .agg(article_count=("title", "count"), average_sentiment=("sentiment_compound", "mean"))
        .reset_index()
        .sort_values("article_count", ascending=False)
    )

    ANALYZED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_reset.to_csv(ANALYZED_CSV, index=False, encoding="utf-8")
    pairs_df.to_csv(PAIRS_CSV, index=False, encoding="utf-8")
    source_summary.to_csv(SOURCE_SUMMARY_CSV, index=False, encoding="utf-8")
    return df_reset, pairs_df, source_summary


def build_network(articles, pairs):
    source_short = {
        "BBC World": "BBC", "Guardian World": "Guardian",
        "NPR World": "NPR", "CNN World": "CNN", "Al Jazeera": "AJ",
    }
    articles = articles.reset_index(drop=True)
    articles["article_id"] = articles.index.map(lambda x: f"article_{x}")

    def make_key(source, title, link):
        return f"{source}|||{title}|||{link}"

    key_to_id = {make_key(r["source"], r["title"], r["link"]): r["article_id"] for _, r in articles.iterrows()}

    net = Network(height="600px", width="100%", bgcolor="#0f1117", font_color="white", notebook=False, cdn_resources="in_line")

    for _, row in articles.iterrows():
        tooltip = (
            f"<b>Source:</b> {row['source']}<br>"
            f"<b>Title:</b> {row['title']}<br>"
            f"<b>Sentiment:</b> {row['sentiment_label']} ({row['sentiment_compound']:.2f})<br>"
            f"<b>Link:</b> {row['link']}"
        )
        color = SOURCE_COLORS.get(row["source"], "#888888")
        label = f"{source_short.get(row['source'], row['source'])} {row.name + 1}"
        net.add_node(row["article_id"], label=label, title=tooltip, color=color, value=12)

    for _, row in pairs.iterrows():
        sim = float(row["similarity"])
        if sim < SIMILARITY_THRESHOLD:
            continue
        k1 = make_key(row["source_1"], row["title_1"], row["link_1"])
        k2 = make_key(row["source_2"], row["title_2"], row["link_2"])
        if k1 not in key_to_id or k2 not in key_to_id:
            continue
        net.add_edge(key_to_id[k1], key_to_id[k2], value=sim * 10, title=f"Similarity: {sim:.3f}", color="#ffffff44")

    NETWORK_HTML.parent.mkdir(parents=True, exist_ok=True)
    net.show_buttons(filter_=["physics"])
    NETWORK_HTML.write_text(net.generate_html(), encoding="utf-8")
    return NETWORK_HTML


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="News Source Comparison",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
.metric-card {
    background: #1a1d2e;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
}
.metric-card .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #f1f5f9;
}
.source-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}
.sentiment-positive { color: #4ade80; }
.sentiment-negative { color: #f87171; }
.sentiment-neutral  { color: #94a3b8; }
.stButton > button {
    width: 100%;
    background: #2d3250;
    color: #e2e8f0;
    border: 1px solid #3d4470;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 10px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #3d4470;
    border-color: #5a6aaa;
}
</style>
""", unsafe_allow_html=True)


# ─────── Sidebar ───────
with st.sidebar:
    st.markdown("## 📰 News Comparison")
    st.markdown("---")

    # Step 1: Collect
    st.markdown("### Step 1 — Fetch News")
    fetch_btn = st.button("🔄 Fetch latest articles")

    if fetch_btn:
        progress_bar = st.progress(0)
        status = st.empty()

        def update_progress(i, source_name):
            progress_bar.progress((i + 1) / len(FEEDS))
            status.markdown(f"Fetching **{source_name}**...")

        with st.spinner(""):
            df_raw = collect_rss(progress_callback=update_progress)
            progress_bar.progress(1.0)
            status.empty()

        st.session_state["df_raw"] = df_raw
        st.session_state["topics_df"] = discover_topics(df_raw)
        st.success(f"✓ {len(df_raw)} articles collected")

    # Step 2: Topic picker
    st.markdown("### Step 2 — Pick a Topic")

    if "df_raw" not in st.session_state and RAW_CSV.exists():
        st.session_state["df_raw"] = pd.read_csv(RAW_CSV)

    if "topics_df" not in st.session_state and TOPICS_CSV.exists():
        st.session_state["topics_df"] = pd.read_csv(TOPICS_CSV)

    topic_options = []
    if "topics_df" in st.session_state:
        topic_options = st.session_state["topics_df"]["term"].head(25).tolist()

    if topic_options:
        suggested = st.selectbox("Suggested topics", ["— choose one —"] + topic_options)
    else:
        suggested = "— choose one —"
        st.caption("Fetch articles first to see suggestions.")

    custom_topic = st.text_input("Or type a custom topic", placeholder='e.g. "climate change"')

    topic = custom_topic.strip() if custom_topic.strip() else (
        suggested if suggested != "— choose one —" else ""
    )

    if topic:
        st.markdown(f"**Selected:** `{topic}`")

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", disabled=(not topic or "df_raw" not in st.session_state))

    if not topic:
        st.caption("Select or type a topic to enable analysis.")
    elif "df_raw" not in st.session_state:
        st.caption("Fetch articles first.")


# ─────── Main area ───────
st.markdown("# News Source Comparison")
st.markdown("Compare how different news outlets cover the same topic — sentiment, similarity, and network.")

if run_btn and topic and "df_raw" in st.session_state:
    with st.spinner(f"Filtering articles for **{topic}**..."):
        filtered = filter_topic(st.session_state["df_raw"], topic)

    if len(filtered) < 2:
        st.error(f"Only {len(filtered)} article(s) found for **{topic}**. Try a different topic.")
        st.stop()

    with st.spinner("Running sentiment analysis and similarity..."):
        articles, pairs, source_summary = analyze(filtered)

    with st.spinner("Building network graph..."):
        network_path = build_network(articles, pairs)

    st.session_state["results"] = {
        "topic": topic,
        "articles": articles,
        "pairs": pairs,
        "source_summary": source_summary,
        "network_path": network_path,
    }

# ─────── Results ───────
if "results" in st.session_state:
    r = st.session_state["results"]
    articles = r["articles"]
    pairs = r["pairs"]
    source_summary = r["source_summary"]
    topic = r["topic"]
    network_path = r["network_path"]

    st.markdown(f"## Results for: `{topic}`")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    total = len(articles)
    sources_n = articles["source"].nunique()
    avg_sent = articles["sentiment_compound"].mean()
    top_pairs = len(pairs[pairs["similarity"] >= SIMILARITY_THRESHOLD]) if len(pairs) > 0 else 0

    with col1:
        st.markdown(f'<div class="metric-card"><div class="label">Articles</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="label">Sources</div><div class="value">{sources_n}</div></div>', unsafe_allow_html=True)
    with col3:
        sent_color = "#4ade80" if avg_sent > 0.05 else ("#f87171" if avg_sent < -0.05 else "#94a3b8")
        st.markdown(f'<div class="metric-card"><div class="label">Avg Sentiment</div><div class="value" style="color:{sent_color}">{avg_sent:+.2f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="label">Similar Pairs</div><div class="value">{top_pairs}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Network", "📊 Source Summary", "📋 Articles", "🔗 Similar Pairs"])

    with tab1:
        st.markdown("Each node is an article. Edges connect articles above the similarity threshold. Hover nodes for details.")
        if network_path.exists():
            html_content = network_path.read_text(encoding="utf-8")
            components.html(html_content, height=650, scrolling=False)
        else:
            st.warning("Network file not found.")

    with tab2:
        st.markdown("### Coverage by source")
        display_summary = source_summary.copy()
        display_summary["average_sentiment"] = display_summary["average_sentiment"].round(3)
        display_summary.columns = ["Source", "Articles", "Avg Sentiment"]
        st.dataframe(display_summary, use_container_width=True, hide_index=True)

        st.markdown("### Sentiment distribution")
        sent_counts = articles["sentiment_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        st.bar_chart(sent_counts.set_index("Sentiment"))

    with tab3:
        st.markdown("### All matching articles")

        source_filter = st.multiselect(
            "Filter by source",
            options=sorted(articles["source"].unique()),
            default=sorted(articles["source"].unique()),
        )
        filtered_view = articles[articles["source"].isin(source_filter)]

        for _, row in filtered_view.iterrows():
            color = SOURCE_COLORS.get(row["source"], "#888")
            sent_class = f"sentiment-{row['sentiment_label']}"
            score = row["sentiment_compound"]
            with st.expander(f"**{row['source']}** — {row['title'][:90]}{'...' if len(row['title']) > 90 else ''}"):
                st.markdown(f"**Published:** {row.get('published', 'N/A')}")
                st.markdown(f"**Sentiment:** <span class='{sent_class}'>{row['sentiment_label']} ({score:+.3f})</span>", unsafe_allow_html=True)
                if row.get("link"):
                    st.markdown(f"[Open article ↗]({row['link']})")

    with tab4:
        st.markdown("### Most similar article pairs")
        if len(pairs) == 0:
            st.info("No pairs to display.")
        else:
            top_pairs_df = pairs[pairs["similarity"] >= SIMILARITY_THRESHOLD].head(20).copy()
            if len(top_pairs_df) == 0:
                st.info(f"No pairs above similarity threshold ({SIMILARITY_THRESHOLD}).")
            else:
                top_pairs_df["similarity"] = top_pairs_df["similarity"].round(3)
                for _, row in top_pairs_df.iterrows():
                    with st.expander(f"**{row['source_1']}** ↔ **{row['source_2']}** — similarity: {row['similarity']}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**{row['source_1']}**")
                            st.markdown(row['title_1'])
                            if row.get('link_1'):
                                st.markdown(f"[Open ↗]({row['link_1']})")
                        with c2:
                            st.markdown(f"**{row['source_2']}**")
                            st.markdown(row['title_2'])
                            if row.get('link_2'):
                                st.markdown(f"[Open ↗]({row['link_2']})")

else:
    st.info("👈 Start by fetching articles in the sidebar, then pick a topic and run the analysis.")