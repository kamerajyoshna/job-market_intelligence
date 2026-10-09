# Project Report: AI-Powered Job Market Intelligence and Skill Gap Analytics Platform

**Bachelor of Technology (B.Tech) in Computer Science & Engineering**  
**Final-Year Capstone Project Documentation**

---

## Abstract
Rapid technological advancement and evolving hiring requirements have created an observable disconnect between university computer science curricula and industrial labor demands. This project presents the design and implementation of an end-to-end, reproducible Data Science, Natural Language Processing (NLP), and Machine Learning platform titled **AI-Powered Job Market Intelligence and Skill Gap Analytics Platform**. The platform ingests unstructured job postings, enforces strict raw-data immutability, normalizes entity schemas, and executes whole-token regular expression phrase extraction across an extensible taxonomy of technical competencies. 

Extracted competencies are synthesized into empirical demand metrics, role-specific cross-tabulations, and skill co-occurrence matrices. An interactive **Skill Gap Analyzer** evaluates user-declared proficiencies against empirical market benchmarks, deriving a demand-weighted readiness score and generating a phased learning roadmap. For resume matching, document vectorization is conducted using sublinear Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity ranking. Unsupervised semantic archetype discovery is implemented using K-Means clustering, evaluated objectively via the Silhouette Coefficient and projected to two dimensions using Truncated Singular Value Decomposition (SVD). The software is developed using Python, Streamlit, Pandas, Scikit-learn, and Plotly, accompanied by an automated suite of 42 unit and integration tests.

---

## 1. Introduction

### 1.1 Background & Motivation
In contemporary recruitment, job descriptions represent rich, unstructured repositories of technical expectations. However, undergraduate students often struggle to identify the exact technical competencies demanded by employers for specific career pathways (such as Data Science, Backend Engineering, or DevOps). Traditional career counseling frequently relies on static or anecdotal advice rather than quantitative data mining.

### 1.2 Problem Statement
Job descriptions are predominantly published in noisy natural language containing HTML tags, variable formatting, inconsistent job titles, ambiguous salary expressions, and unstructured skill lists. There is a critical need for an automated, lightweight, transparent analytical system capable of:
1. Ingesting diverse job vacancy datasets with schema variability.
2. Extracting technical competencies accurately without expensive commercial APIs.
3. Conducting rigorous, non-fabricating compensation and demand analysis.
4. Providing transparent, actionable skill-gap recommendations.

### 1.3 Project Objectives
- **Data Engineering:** Build an ingestion and cleaning pipeline supporting CSV datasets with automated column detection, missing value auditing, and duplicate pruning.
- **NLP Information Extraction:** Implement an extensible, regex-based skill extractor with token boundary enforcement to prevent substring false positives.
- **Labor Market Analytics:** Calculate empirical skill frequencies, role-wise demand heatmaps, and skill co-occurrence synergy matrices.
- **Skill Gap Modeling:** Develop a transparent mathematical formulation for career readiness scoring and prioritized milestone roadmapping.
- **Information Retrieval & ML:** Implement TF-IDF vector space modeling for document matching (cosine similarity) and K-Means clustering with Silhouette score optimization.
- **Verification:** Establish an automated Pytest test suite covering data integrity, edge cases, algorithmic formulas, and headless UI execution.

### 1.4 Scope & Boundaries
- The project runs on standard consumer hardware without requiring GPUs or commercial API keys.
- Similarity scores represent textual overlap in vector space and explicitly do NOT represent hiring probabilities.
- Salary analysis strictly isolates currencies (no artificial foreign exchange conversions).

---

## 2. Literature Review & Theoretical Foundation

### 2.1 Natural Language Processing in Labor Market Intelligence
Labor Market Intelligence (LMI) has increasingly shifted from manual surveys to automated scraping and NLP parsing. Existing literature identifies three primary extraction paradigms:
1. **Dictionary & Rule-Based Matchers:** Utilize curated skill taxonomies and regular expression token matching. Advantages include high precision, transparency, zero hallucination, and instantaneous CPU inference.
2. **Named Entity Recognition (NER) via Statistical Models (CRF / BiLSTM-CRF):** Capable of discovering uncatalogued entities but requires thousands of manually annotated domain tokens.
3. **Transformer-based Large Language Models (LLMs):** Powerful contextual understanding, but introduce non-deterministic outputs, API costs, latency, and potential hallucination.
*Design Decision:* For an academic final-year project prioritizing reproducibility and zero financial dependencies, a compiled regex engine with token boundary lookarounds provides the optimal trade-off of speed, determinism, and explainability.

### 2.2 Vector Space Models & TF-IDF
Salton and Buckley (1988) formulated the Vector Space Model (VSM) for information retrieval. Term Frequency-Inverse Document Frequency (TF-IDF) penalizes ubiquitously occurring stop words while elevating discriminative domain terms:
$$\text{TF-IDF}(t, d, D) = (1 + \log(\text{TF}(t, d))) \times \left(\log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1\right)$$
Sublinear term frequency scaling dampens the effect of repetitive keywords, preventing keyword stuffing from dominating retrieval.

### 2.3 Document Similarity via Cosine Distance
Cosine similarity computes the angle between query vector $\mathbf{q}$ and document vector $\mathbf{d}$, normalizing for text length:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$

### 2.4 Unsupervised Clustering & Validation
K-Means partitions $N$ document vectors into $k$ clusters by minimizing the Within-Cluster Sum of Squares (WCSS). Because unsupervised text data lacks ground-truth class labels, supervised metrics (e.g. Accuracy, F1-score) cannot be used. Cluster cohesion and separation are objectively measured via Rousseeuw's (1987) Silhouette Coefficient:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, 1]$$

---

## 3. System Architecture & Methodology

### 3.1 Architectural Diagram
```
+-----------------------------------------------------------------------------+
|                                DATA LAYER                                   |
|   Uploaded CSV / Synthetic Fallback  -->  data_loader.py (Column Detection) |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                           PREPROCESSING LAYER                               |
|   clean_dataframe()  -->  HTML Stripping, Case Normalization, Title RegEx   |
|   parse_salary()     -->  Currency & Period Extraction (No FX conversion)  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                              NLP & ML CORE                                  |
|   skill_extractor.py  -->  Token-Boundary RegEx (?<![a-z0-9+#])skill...     |
|   demand_analysis.py  -->  Frequency, Group Heatmap, Co-occurrence Matrix   |
|   salary_analysis.py  -->  Stratified Statistics, Association Differentials |
|   skill_gap.py        -->  Demand-Weighted Readiness Score & Phased Roadmap |
|   job_similarity.py   -->  Sublinear TF-IDF + Cosine Similarity Matcher     |
|   clustering.py       -->  K-Means Clustering + Silhouette + TruncatedSVD    |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                          PRESENTATION LAYER (UI)                            |
|             Streamlit Multi-Page Interactive Dashboard (9 Pages)            |
|                   Plotly Visualizations & Metric Indicators                 |
+-----------------------------------------------------------------------------+
```

### 3.2 Key Algorithmic Modules

#### 3.2.1 Token Boundary Skill Extraction (`src/skill_extractor.py`)
To prevent false-positive token substring matches (e.g. "java" matching inside "javascript", or "go" matching inside "good"), each skill and its aliases compile to a specialized regex:
```python
r"(?<![a-z0-9+#])(?:" + "|".join(escaped_terms) + r")(?![a-z0-9+#])"
```
This enables recognition of special symbols common in computer science (`c++`, `c#`, `node.js`, `ci/cd`, `a/b testing`).

#### 3.2.2 Transparent Skill Gap Formulation (`src/skill_gap.py`)
Given target role $R$, let $T$ be the set of top-$N$ skills where empirical demand $\text{DemandPct}(s) \ge 5\%$.
Let $U$ be the candidate's canonical skill set.
- **Matched Skills:** $M = T \cap U$
- **Missing Skills:** $G = T \setminus U$
- **Demand-Weighted Readiness Score:**
$$\text{Readiness Score} = \left(\frac{\sum_{s \in M} \text{DemandPct}(s)}{\sum_{s \in T} \text{DemandPct}(s)}\right) \times 100$$
- **Priority Tier Allocation:**
  - **Phase 1 (Core Non-Negotiables):** $\text{DemandPct}(s) \ge 50\%$
  - **Phase 2 (Profile Strengtheners):** $25\% \le \text{DemandPct}(s) < 50\%$
  - **Phase 3 (Differentiators):** $\text{DemandPct}(s) < 25\%$

---

## 4. Dataset Management & Integrity Standards

### 4.1 Schema Definition
The system supports both real and synthetic CSV datasets with the following fields:
- `title` (Mandatory): Job posting designation.
- `description` / `skills` (At least one mandatory): Unstructured job description text or skill requirements.
- `company` (Optional): Employer or hiring organization.
- `location` (Optional): Work location or remote indicator.
- `salary` (Optional): Textual or numeric compensation string.
- `date` (Optional): Posting timestamp.

### 4.2 Raw Data Immutability
All ingestion and cleaning operations return a newly instantiated Pandas DataFrame. The original raw data remains unaltered in session memory to guarantee reproducibility and auditability.

### 4.3 Demonstration Fallback Dataset
When an external CSV is not loaded, the application provides a 600-row synthetic demonstration dataset (`data/sample_synthetic_jobs.csv`) generated deterministically using a fixed random seed (`RANDOM_SEED = 42`). Every row is tagged with `data_source = "SYNTHETIC (generated, not real market data)"`.

---

## 5. Implementation Details

- **Programming Language:** Python 3.10+
- **Frontend / Dashboard:** Streamlit 1.65 (Native layouts, tabs, expanders, forms)
- **Data Manipulation:** Pandas 3.0, NumPy 2.5
- **NLP & Machine Learning:** Scikit-learn 1.9 (`TfidfVectorizer`, `KMeans`, `TruncatedSVD`, `silhouette_score`, `cosine_similarity`)
- **Data Visualization:** Plotly 7.1 (`plotly.express` and `plotly.graph_objects`)
- **Automated Testing:** Pytest 9.1 with `streamlit.testing.v1.AppTest`

### Directory Layout
```
job-market-intelligence/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Pinned compatible dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Setup and execution guide
├── data/
│   ├── README.md               # Dataset documentation & attribution
│   └── sample_synthetic_jobs.csv
├── src/
│   ├── __init__.py
│   ├── config.py               # Constants, random seed, skill dictionary, title rules
│   ├── data_loader.py          # CSV loading, column synonym detection, synthetic generator
│   ├── preprocessing.py        # Text normalization, salary parsing, title rules, data quality
│   ├── skill_extractor.py      # Token-boundary regex skill extraction engine
│   ├── demand_analysis.py      # Frequency, cross-role heatmaps, co-occurrence, trend analysis
│   ├── salary_analysis.py      # Currency-isolated compensation analytics
│   ├── skill_gap.py            # Demand-weighted readiness and phased roadmap
│   ├── job_similarity.py       # TF-IDF vectorizer + cosine similarity matcher
│   ├── clustering.py           # K-Means clustering + silhouette evaluation + SVD projection
│   └── pipeline.py             # End-to-end transformation pipeline
├── tests/
│   ├── __init__.py
│   ├── test_analytics.py       # Tests for demand, salary, gap, similarity, and clustering
│   ├── test_preprocessing.py   # Tests for cleaning, salary parsing, deduplication, errors
│   ├── test_skill_extractor.py # Tests for token boundary, aliases, case, and extensibility
│   └── test_app_ui.py          # Headless automated UI verification via Streamlit AppTest
└── docs/
    ├── methodology.md          # Technical methodology and mathematical formulations
    ├── project_report.md       # Full academic project documentation report
    ├── viva_questions.md       # Comprehensive viva voce Q&A defense guide
    ├── deployment.md           # Deployment manual (Streamlit Cloud, Render, HuggingFace)
    └── demo_walkthrough.md     # 10-minute viva presentation script
```

---

## 6. Verification & Testing

The project incorporates 42 automated tests executed via `pytest`:
- **Preprocessing Tests (`test_preprocessing.py`):** Verify HTML stripping, salary parsing across USD, INR, GBP, EUR, monthly/hourly multipliers, title classification, column synonym auto-mapping, deduplication, and error handling.
- **Skill Extractor Tests (`test_skill_extractor.py`):** Verify multi-word skills, case insensitivity, alias mapping, token boundary protection (ensuring "java" does not trigger on "javascript", and "c++" / "node.js" match correctly), and dynamic taxonomy extension.
- **Analytics & ML Tests (`test_analytics.py`):** Verify frequency calculation, symmetric co-occurrence matrices, intra-currency salary isolation, transparent gap mathematical formulas, TF-IDF cosine ranking order, K-Means clustering reproducibility, and small dataset graceful skip handling.
- **Headless UI Tests (`test_app_ui.py`):** Simulate user interaction via `streamlit.testing.v1.AppTest`, verifying that all 9 dashboard pages render with 0 runtime exceptions.

---

## 7. Limitations & Ethical Considerations

1. **Taxonomy Completeness:** Dictionary-based matching cannot detect emerging frameworks or niche libraries not catalogued in the dictionary.
2. **Context & Negation:** Pure keyword matching cannot distinguish between required competencies and nice-to-have preferences, nor can it detect negative expressions (e.g., "no PHP required").
3. **Dataset Sampling Biases:** Results strictly reflect the geographic and domain distribution of the uploaded dataset.
4. **Salary & Association Disclaimers:** Skill-salary differentials represent empirical statistical associations, not direct causal wage premiums.
5. **No Hiring Prediction:** Cosine similarity scores quantify lexical vector overlap and must never be interpreted as hiring probabilities.

---

## 8. Conclusion & Future Enhancements

The **AI-Powered Job Market Intelligence and Skill Gap Analytics Platform** establishes a robust, mathematically grounded, and reproducible tool for labor market analysis and student career readiness evaluation. By avoiding paid APIs and black-box models, the project demonstrates sound data engineering, transparent NLP information extraction, and rigorous unsupervised machine learning principles suitable for final-year engineering evaluation.

### Future Scope:
- Integration of Contextual Embeddings (Sentence-BERT) and Spacy PhraseMatcher for hybrid semantic entity extraction.
- Parsing of candidate PDF / DOCX resumes via optical and layout-aware document parsers.
- Supervised compensation regression models incorporating cross-validated MAE and $R^2$ evaluations.
- Temporal forecasting using ARIMA / Prophet for longitudinal skill demand trajectory tracking.

---

## 9. Academic References

1. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513-523.
2. Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53-65.
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
5. Tambe, P. (2014). *Big Data Investment, Skills, and Firm Value*. Management Science, 60(6), 1452-1469.
