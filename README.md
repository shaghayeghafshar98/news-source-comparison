# News Source Comparison

Collects recent world news from five RSS feeds, filters articles by a topic of your choice, and compares how different sources cover it — including sentiment analysis, text similarity, an interactive network graph, and an HTML report.

**Sources:** BBC World · The Guardian · NPR · CNN · Al Jazeera

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

Run the five steps in order. Replace `"Iran"` with any topic you want.

### 1. Collect articles from RSS feeds

```bash
python src/collect_rss.py
```

Saves raw articles to `data/raw/all_rss_articles.csv`.

### 2. (Optional) Discover trending topics

```bash
python src/discover_topics.py
```

Prints the top terms and phrases across all collected articles — useful for picking a topic. Saves results to `data/processed/discovered_topics.csv`.

### 3. Filter by topic

```bash
python src/filter_topic.py --topic "Iran"
```

Multi-word topics are supported (all words must appear in the article):

```bash
python src/filter_topic.py --topic "climate change"
```

Saves matching articles to `data/processed/topic_articles.csv`.

### 4. Analyze sentiment and similarity

```bash
python src/compare_articles.py
```

Runs VADER sentiment scoring and TF-IDF cosine similarity on all filtered articles. Saves three files to `data/processed/`.

### 5. Build the similarity network

```bash
python src/build_network.py
```

Generates an interactive network graph at `outputs/networks/news_similarity_network.html`. Each node is an article; edges connect articles above the similarity threshold. Open the file in a browser to explore.

### 6. Generate the HTML report

```bash
python src/generate_report.py --topic "Iran"
```

Produces `outputs/reports/news_comparison_report.html` with a source summary table, article index, most-similar pairs, and a link to the network visualization.

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
| `outputs/reports/news_comparison_report.html` | Full HTML report |