# AI-Powered Job Market Intelligence and Skill Gap Analytics Platform Using NLP & Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Data Science, Natural Language Processing, and Machine Learning web application designed to analyze job market postings, extract technical competencies, model labor demand distributions, evaluate personal skill gaps, match candidate profiles, and discover latent job archetypes using unsupervised clustering.

Built specifically for **Final-Year Computer Science & Engineering (CSE) Capstone Projects**, technical viva voce reviews, and software engineering portfolios.

---

## 🌟 Key Features

1. **Robust Dataset Ingestion & Validation (`src/data_loader.py`):**
   - Supports any standard job vacancy CSV upload with automatic column synonym mapping (`title`, `company`, `description`, `skills`, `location`, `salary`, `date`).
   - Strict raw-data immutability with comprehensive data quality auditing (missing value rates, deduplication, and dropped record accounting).
   - Bundled deterministic 600-row synthetic demonstration dataset (`seed=42`) with clear demonstration labeling.

2. **NLP Skill Extraction Engine (`src/skill_extractor.py`):**
   - Whole-token regular expression matcher with lookaround token boundaries `(?<![a-z0-9+#])skill(?![a-z0-9+#])` preventing substring false positives (e.g. `java` vs `javascript`, `c++`, `node.js`, `ci/cd`).
   - Supports multi-word compound phrases and technical aliases (e.g., `ML` $\rightarrow$ `machine learning`, `sklearn` $\rightarrow$ `scikit-learn`).
   - Dynamically extensible taxonomy via interactive UI or central configuration (`src/config.py`).

3. **Labor Demand Analytics (`src/demand_analysis.py`):**
   - Empirical skill demand frequency (% of postings requiring skill).
   - Cross-role skill intensity heatmaps and employer distribution metrics.
   - Skill co-occurrence synergy matrix displaying technologies frequently demanded together.

4. **Curated Salary Intelligence (`src/salary_analysis.py`):**
   - Parses multi-currency compensation strings (USD, INR, GBP, EUR) normalized to annual equivalents (hourly $\times 2080$, monthly $\times 12$).
   - Strict intra-currency isolation: **never converts or combines disparate currencies** across artificial exchange rates.
   - Median differential skill-salary association modeling with transparent causal disclaimers.

5. **Student Skill Gap Analyzer & Phased Roadmap (`src/skill_gap.py`):**
   - 1-click candidate profile presets (Data Scientist, Web Developer, Data Analyst, Cloud/DevOps).
   - Computes a mathematical **Demand-Weighted Readiness Score**.
   - Categorizes missing competencies into empirical priority milestones (Phase 1: Core Non-Negotiables $\ge 50\%$, Phase 2: Strengtheners $25\% - 50\%$, Phase 3: Differentiators $< 25\%$).

6. **TF-IDF Vector Space Job Matching (`src/job_similarity.py`):**
   - Sublinear Term Frequency-Inverse Document Frequency (TF-IDF) with n-grams (1, 2) and cosine similarity ranking.
   - 1-click sample resume profiles for instant demonstration.
   - Highlights shared technical skills between candidate query and retrieved postings.
   - Explicit disclaimer: measures textual vector similarity, NOT hiring probability.

7. **Unsupervised Machine Learning Clustering (`src/clustering.py`):**
   - High-dimensional K-Means clustering on TF-IDF document vectors.
   - Optimal cluster count ($k$) heuristic evaluation via the **Silhouette Coefficient** ($s \in [-1, 1]$).
   - 2D dimensionality reduction via Truncated Singular Value Decomposition (SVD) for interactive Plotly scatter visualization.
   - Representative keyword extraction and cluster-vs-role contingency cross-tabulation.

---

## 🏗️ Architecture & Data Flow

```
[Raw CSV Dataset / Synthetic Fallback]
                  │
                  ▼
         data_loader.py (Column Synonym Auto-Mapping)
                  │
                  ▼
        preprocessing.py (HTML Stripping, Salary Parsing, Deduplication)
                  │
                  ▼
        skill_extractor.py (Token-Boundary RegEx NLP Engine)
                  │
  ┌───────────────┼───────────────┬────────────────┬──────────────┐
  ▼               ▼               ▼                ▼              ▼
demand_       salary_          skill_            job_         clustering.py
analysis.py   analysis.py      gap.py        similarity.py   (K-Means + SVD)
(Demand/Cooc) (Intra-Currency) (Readiness/   (TF-IDF +
                                Roadmap)     Cosine Sim)
  │               │               │                │              │
  └───────────────┴───────────────┼────────────────┴──────────────┘
                                  ▼
                   app.py (Streamlit Web Dashboard)
```

---

## 🚀 Quickstart & Installation (Windows PowerShell)

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed on your system.

### Step 1: Clone or Navigate to the Workspace
```powershell
cd c:\Users\kamer\Downloads\job-market-intelligence
```

### Step 2: Set Up Virtual Environment & Activate
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled on Windows, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing Suite

The project includes **42 automated unit and integration tests** verifying data cleaning, edge cases, algorithmic formulas, and headless UI rendering:

```powershell
pytest -v
```

### Test Coverage Highlights:
- `tests/test_preprocessing.py`: Text cleaning, salary parsing across USD/INR/GBP/EUR, negative/invalid strings, title normalization, column synonym detection, deduplication, and empty dataset handling.
- `tests/test_skill_extractor.py`: Multi-word skills, alias mapping, case insensitivity, regex token boundary protection ("java" vs "javascript", "c++", "node.js"), and taxonomy extension.
- `tests/test_analytics.py`: Demand frequencies, symmetric co-occurrence matrices, intra-currency salary isolation, weighted readiness formulas, cosine similarity monotonic ordering, and K-Means reproducibility.
- `tests/test_app_ui.py`: Headless full-stack verification using `streamlit.testing.v1.AppTest`, asserting all 9 dashboard pages load with 0 exceptions.

---

## 📁 Project Directory Layout

```
job-market-intelligence/
├── app.py                      # Streamlit multi-page web application (9 pages)
├── requirements.txt            # Dependency manifest (streamlit, pandas, scikit-learn, etc.)
├── pytest.ini                  # Pytest configuration settings
├── README.md                   # Complete project reference and execution guide
├── data/
│   ├── README.md               # Dataset ethics, licensing, and attribution guide
│   └── sample_synthetic_jobs.csv  # 600-row balanced demonstration dataset (seed=42)
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants, random seed, skill dictionary, title rules
│   ├── data_loader.py          # CSV loader, column synonym detection, synthetic generator
│   ├── preprocessing.py        # Text cleaning, salary parser, deduplication, quality report
│   ├── skill_extractor.py      # Token-boundary regex skill extraction engine
│   ├── demand_analysis.py      # Demand frequency, cross-role heatmap, co-occurrence, trends
│   ├── salary_analysis.py      # Currency-isolated compensation analytics
│   ├── skill_gap.py            # Demand-weighted readiness index and phased roadmap
│   ├── job_similarity.py       # Sublinear TF-IDF vectorizer + cosine similarity matcher
│   ├── clustering.py           # K-Means clustering + silhouette optimization + SVD projection
│   └── pipeline.py             # End-to-end data processing pipeline
├── tests/
│   ├── __init__.py
│   ├── test_analytics.py       # Analytics, salary, gap, similarity, and clustering tests
│   ├── test_preprocessing.py   # Data cleaning, salary parser, and column detection tests
│   ├── test_skill_extractor.py # Regex boundary, alias, and taxonomy tests
│   └── test_app_ui.py          # Automated headless Streamlit AppTest suite
└── docs/
    ├── methodology.md          # Technical methodology and mathematical formulations
    ├── project_report.md       # Comprehensive academic final-year project report
    ├── viva_questions.md       # 20 in-depth viva voce defense questions and answers
    ├── deployment.md           # Free deployment manual (Streamlit Cloud, Render, HuggingFace)
    └── demo_walkthrough.md     # 10-minute viva evaluation walkthrough script
```

---

## 🎓 Academic Viva Voce Quick Reference

1. **Why use a dictionary-based regex extractor instead of a deep learning model?**  
   It is deterministic, explainable, lightweight (CPU-only, no GPUs), requires no expensive annotated token training data, and eliminates hallucinations.
2. **How does the extractor avoid the "Java" vs "JavaScript" trap?**  
   It utilizes negative lookaround regex boundaries `(?<![a-z0-9+#])` and `(?![a-z0-9+#])`, sorting aliases by length descending so compound phrases match first.
3. **Why is Accuracy not reported for the ML models?**  
   K-Means and TF-IDF similarity are **unsupervised algorithms** operating on unlabeled text. Reporting classification accuracy would be fabricated. We evaluate clustering using the **Silhouette Coefficient** ($s \in [-1, 1]$).
4. **Why are currencies never converted across exchange rates?**  
   Converting wages using nominal exchange rates distorts purchasing power parity and local economic context. Currencies are strictly isolated.
5. **Does cosine similarity indicate hiring probability?**  
   No. It measures lexical angle in vector space, not candidate competency or hiring probability.

*Full 20-question viva defense guide available in [docs/viva_questions.md](docs/viva_questions.md).*

---

## 🌐 Free Cloud Deployment

### Deploy to Streamlit Community Cloud (Free)
1. Push this repository to GitHub:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: AI Job Market Intelligence Platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/job-market-intelligence.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**, select your repository, set the main file path to `app.py`, and click **Deploy**.
4. The application will build and deploy online for free with a public URL.

*Detailed deployment steps for Hugging Face Spaces and Render are documented in [docs/deployment.md](docs/deployment.md).*

---

## ⚖️ Ethics, Licensing & Limitations

- **Synthetic Data Disclaimer:** The bundled dataset `sample_synthetic_jobs.csv` is synthetic and generated solely for algorithm verification and academic presentation. It does not reflect live labor market trends.
- **Real Datasets:** When loading public datasets (e.g. from Kaggle or open government portals), verify that access and attribution adhere to the dataset's specific license.
- **Code License:** Distributed under the MIT License.
