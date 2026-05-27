# News Source Comparison

Collects recent world news from five RSS feeds, filters articles by a topic of your choice, and compares how different sources cover it — including sentiment analysis, text similarity, an interactive network graph, and an HTML report.

**Sources:** BBC World · The Guardian · NPR · CNN · Al Jazeera

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Usage

1. Click **"Fetch latest articles"** in the sidebar — pulls fresh articles from all 5 sources
2. A dropdown of **suggested topics** appears automatically based on what's trending
3. Pick one or type your own custom topic
4. Click **"Run Analysis"** — the full pipeline runs in one go
5. Explore the results across 4 tabs:
   - 🌐 **Network** — interactive graph showing article similarity across sources
   - 📊 **Source Summary** — per-source article count and average sentiment
   - 📋 **Articles** — filterable list with sentiment scores and links
   - 🔗 **Similar Pairs** — most similar article pairs across different outlets

---

## CLI usage (optional)

The individual scripts still work as standalone tools if needed:

```bash
python src/collect_rss.py
python src/discover_topics.py
python src/filter_topic.py --topic "Ukraine"
python src/compare_articles.py
python src/build_network.py
python src/generate_report.py --topic "Ukraine"
```

---

## Configuration

| File | Setting | Default | Description |
|---|---|---|---|
| `src/build_network.py` | `SIMILARITY_THRESHOLD` | `0.30` | Minimum cosine similarity to draw an edge. Lower = denser graph. |

---

## Output files

| Path | Description |
|---|---|
| `data/raw/all_rss_articles.csv` | All collected articles |
| `data/processed/topic_articles.csv` | Articles matching the chosen topic |
| `data/processed/topic_articles_analyzed.csv` | Articles with sentiment scores |
| `data/processed/similarity_pairs.csv` | All article pairs with similarity scores |
| `data/processed/source_summary.csv` | Per-source article count and average sentiment |
| `outputs/networks/news_similarity_network.html` | Interactive network graph |
| `outputs/reports/news_comparison_report.html` | Full HTML report (CLI only) |