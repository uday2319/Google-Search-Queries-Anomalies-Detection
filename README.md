<div align="center">

# 🔍 Google Search Queries — Anomaly Detection

**Finding the search queries that don't behave like the rest. This project explores Google Search Console query data, then uses an Isolation Forest to flag queries with unusual clicks, impressions, CTR or ranking position.**
 

![Algorithm](https://img.shields.io/badge/Algorithm-Isolation%20Forest-success?style=flat-square)
![Learning](https://img.shields.io/badge/Learning-Unsupervised-blue?style=flat-square)
![Contamination](https://img.shields.io/badge/Contamination-1%25-orange?style=flat-square)

</div>

---

## 📌 Overview

Website owners get a report from Google Search Console that lists every search query that showed their site. For each query it gives the **clicks**, **impressions**, **click-through rate (CTR)** and **average ranking position**. With hundreds of queries, the interesting ones are hard to spot by eye: a query that gets a lot of impressions but almost no clicks, or one that ranks oddly well.

This project does two things:

1. **Exploratory analysis.** It finds the most common words across all queries, the top queries by clicks, impressions and CTR, and how the metrics relate to each other.
2. **Anomaly detection.** It trains an **Isolation Forest**, an unsupervised model, to flag the roughly 1% of queries whose combination of metrics is most unusual. No labelled data is needed.

---

## 📂 Dataset

The data is a **Queries.csv** export from Google Search Console (Performance → Search results → Queries → Export).

| Column        | Description                                                  |
|---------------|--------------------------------------------------------------|
| `Top queries` | The search text people typed into Google                     |
| `Clicks`      | Number of clicks the site received from that query           |
| `Impressions` | Number of times the site appeared in results for that query  |
| `CTR`         | Click-through rate (`Clicks / Impressions`), exported as `"x.xx%"` |
| `Position`    | Average ranking position of the site for that query          |

> You can use your own site's Search Console export. Any CSV with these columns will work.

---

## ⚙️ Workflow

```
Queries.csv
   │
   ├─► 1. Preprocessing ──── convert CTR from "12.5%" text to a 0.125 float
   │
   ├─► 2. Text analysis ──── tokenise queries with regex → top 20 most frequent words
   │
   ├─► 3. Performance EDA ── top 10 queries by Clicks / Impressions
   │                         top 10 & bottom 10 queries by CTR
   │
   ├─► 4. Correlation ────── heatmap of Clicks, Impressions, CTR, Position
   │
   └─► 5. Anomaly detection ─ Isolation Forest on [Clicks, Impressions, CTR, Position]
                              → scatter plot of normal vs anomalous queries
```

### 1. Preprocessing
`CTR` is stored as a percentage string, so it is stripped of `%` and converted to a decimal float so it can be used numerically.

### 2. Query word frequency
Each query is lowercased and split into alphabetic words with the regex `\b[a-zA-Z]+\b`. A `Counter` collects word counts across all queries, and the **20 most common words** are shown in a Plotly bar chart. This shows which topics bring the most search traffic.

### 3. Query performance
Interactive Plotly bar charts show:
- 🔝 Top 10 queries by **Clicks**
- 👀 Top 10 queries by **Impressions**
- 📈 Top 10 queries by **CTR**
- 📉 Bottom 10 queries by **CTR**, which are candidates for better titles or meta descriptions

### 4. Correlation analysis
A correlation heatmap (`px.imshow`) shows how the four metrics relate. For example, it shows whether a better (lower) position goes with a higher CTR.

### 5. Anomaly detection with Isolation Forest

```python
from sklearn.ensemble import IsolationForest

features = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']]

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(features)

queries_df['anomaly'] = iso_forest.predict(features)   # -1 = anomaly, 1 = normal
anomalies = queries_df[queries_df['anomaly'] == -1]
```

**Why Isolation Forest?**
- It is **unsupervised**, so no "this query is abnormal" labels are needed.
- It isolates points by making random splits. Unusual points get separated in fewer splits, so they have shorter path lengths.
- It is fast, scales well, and handles features on very different scales, such as raw impressions next to CTR ratios.

| Parameter       | Value | Meaning                                         |
|-----------------|-------|-------------------------------------------------|
| `n_estimators`  | 100   | Number of isolation trees in the ensemble       |
| `contamination` | 0.01  | Expected share of anomalies (about 1% of queries) |

The results are plotted as an **Impressions vs Clicks** scatter plot with normal queries in blue and anomalies in red.

---

## 💡 What the anomalies tell you

The flagged queries usually fall into a few groups:

- **High impressions, very low clicks.** The page is seen but not chosen, so the title or snippet may need work.
- **Unusually high CTR.** These are strong performers whose title or content could be reused elsewhere.
- **Very high traffic outliers.** These are brand or main queries that make up a large share of traffic.
- **Odd position vs CTR combinations.** For example, a query that ranks well but gets few clicks, which may point to rich results or a mismatch with what the searcher wanted.

These are useful starting points for **SEO decisions** and **content strategy**.

---

## 🛠️ Tech Stack

| Tool             | Purpose                                         |
|------------------|-------------------------------------------------|
| **Python**       | Core language                                   |
| **Pandas**       | Data loading and cleaning                       |
| **re / Counter** | Query tokenisation and word frequency           |
| **Plotly**       | Interactive bar charts and correlation heatmap  |
| **Matplotlib**   | Anomaly scatter plot                            |
| **scikit-learn** | Isolation Forest anomaly detection              |
| **Jupyter**      | Notebook environment                            |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/uday2319/Google-Search-Queries-Anomalies-Detection.git
cd Google-Search-Queries-Anomalies-Detection
```

### 2. Install dependencies
```bash
pip install pandas plotly matplotlib scikit-learn jupyter
```

### 3. Add the dataset
Put `Queries.csv` in the project folder and update the path in the notebook:
```python
queries_df = pd.read_csv("Queries.csv")
```

### 4. Run the notebook
```bash
jupyter notebook Googlesearchproject1.ipynb
```

---

## 📁 Project Structure

```
Google-Search-Queries-Anomalies-Detection/
├── Googlesearchproject1.ipynb   # Full analysis + anomaly detection notebook
├── Queries.csv                  # Google Search Console query export
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Scale features (`StandardScaler`) before Isolation Forest and compare results
- [ ] Use `decision_function()` anomaly scores to rank queries by how unusual they are
- [ ] Compare with other detectors: **Local Outlier Factor**, **One-Class SVM**, **DBSCAN**
- [ ] Add date-wise Search Console data for **time-series** spike detection
- [ ] Build a **Streamlit dashboard** where users upload their own CSV and see anomalies

---

## 👤 Author

**Uday**
