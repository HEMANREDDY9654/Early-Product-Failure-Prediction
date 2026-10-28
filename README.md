# Early Product Failure Prediction from Review Evolution

**Subtitle:** AI-Based Early Warning System for Emerging Product Problems  
**Project Category:** AI/ML Mini-Project (2 Credits)  
**Author / Pair Programming:** Academic Research Implementation  

---

## 1. Project Overview

Modern e-commerce platforms (such as Amazon, Flipkart, Best Buy) aggregate thousands of customer reviews daily. While standard sentiment analysis can determine whether an individual review is positive or negative, it does not detect when multiple specific failure symptoms co-occur and escalate over time.

This project implements an **Early Product Failure Prediction System** that analyzes time-stamped e-commerce customer reviews to provide proactive early warnings for emerging product defects before widespread failures occur.

---

## 2. Problem Statement & Research Question

### Problem Statement
Standard customer sentiment analytics aggregates overall star ratings and polarity distributions. However, a product might maintain an overall 4-star rating while experiencing a sharp, localized escalation in critical subsystem complaints—such as battery drain, overheating, and charging failures occurring together after a firmware update or manufacturing defect. 

Detecting these patterns requires tracking the **temporal co-evolution of multiple complaint signals**, calculating transparent risk trajectories, and detecting abnormal baseline shifts.

### Central Research Question
> *"Can the temporal co-evolution of multiple complaint signals provide an early warning of a potential product failure?"*

### Objectives
1. Build an end-to-end reproducible pipeline from raw customer reviews to explainable alerts.
2. Formulate a multi-signal product risk scoring model integrating frequency, severity, growth rate, symptom diversity, and arrival velocity.
3. Formulate EWMA and rolling baseline deviation anomaly detection without future-data leakage.
4. Compare the proposed multi-signal approach against a simple frequency baseline.
5. Provide a Streamlit dashboard with interactive exploration, evaluation, and explainable alerts.

---

## 3. Main Contribution

The core contribution is a **product-centric, temporal, multi-signal risk assessment framework**. Rather than relying on review volume or overall sentiment in isolation, the system models product risk as the synthesis of five distinct signals:

1. **Complaint Frequency ($\text{proportion}$)**: Proportion of reviews mentioning specific complaint aspects.
2. **Complaint Severity ($\text{severity}$)**: Negative intensity quantified through transformer sentiment combined with star ratings.
3. **Complaint Growth Rate ($\text{growth}$)**: Period-over-period percentage acceleration of complaints relative to past baselines.
4. **Symptom Diversity ($\text{diversity}$)**: Breadth of distinct subsystem failure aspects active in the same period.
5. **Review Arrival Velocity ($\text{velocity}$)**: Sudden surges in review submission volume relative to historical moving averages.

---

## 4. System Architecture & Workflow

### Architecture Pipeline
```
Customer Reviews (CSV)
         │
         ▼
[Data Ingestion & Validation]
  ├── Schema validation & Column auto-mapping
  ├── Robust date parsing & Unix timestamp conversion
  └── Deduplication & Missing-data handling
         │
         ▼
[Aspect & Complaint Extraction]
  ├── Multi-aspect keyword/phrase lexicon matcher
  └── Unclassified negative review detection ("Other")
         │
         ▼
[Sentiment & Severity Analysis]
  ├── Pretrained Transformer (DistilBERT SST-2)
  ├── Rule-based VADER-style offline fallback
  └── Rating-augmented severity score (0 to 1)
         │
         ▼
[Temporal Signal Engineering]
  ├── Weekly / Monthly chronological aggregation
  ├── Per-aspect and overall proportions
  ├── Rolling EWMA historical baseline (no future leakage)
  └── Growth rates and velocity ratios
         │
         ▼
[Product Risk Scoring & Anomaly Detection]
  ├── Multi-signal normalized weighted combination (0 to 100)
  ├── Rolling baseline z-score deviation thresholding
  └── Risk band classification (Low, Moderate, High, Critical)
         │
         ▼
[Explainable Alerts & Comparative Evaluation]
  ├── Transparent evidence generation from computed metrics
  ├── Frequency baseline comparison
  └── Ablation analysis across all 5 signals
         │
         ▼
[Interactive Streamlit Dashboard]
```

---

## 5. Mathematical Formulations & Feature Definitions

All features are calculated on chronologically ordered periods $t \in \{1, 2, \dots, T\}$. Let $R_t$ be the set of valid reviews received in period $t$, and let $A$ be the set of supported complaint aspects ($|A| = 9$).

### A. Overall Complaint-Review Proportion
The proportion of valid reviews in period $t$ that contain at least one complaint (strictly bounded $\le 100\%$):
$$p_t = \frac{|\{r \in R_t : \text{is\_complaint}(r) = \text{True}\}|}{|R_t|}$$
*Crucial Design Rule:* A review mentioning multiple complaint aspects (e.g. Battery + Heating + Charging) is counted **exactly once** in $p_t$. The denominator $|R_t|$ is the count of total valid reviews retained in period $t$.

### B. Aspect-Specific Complaint Proportion
For each aspect $a \in A$ in period $t$, the proportion of valid reviews mentioning aspect $a$ (bounded $\le 100\%$):
$$\text{proportion}(a, t) = \frac{|\{r \in R_t : a \in \text{aspects}(r)\}|}{|R_t|}$$

### C. Aspect Mention Density (Multi-Symptom Co-occurrence)
The average number of failure symptoms reported per review across all aspects in period $t$:
$$m_t = \frac{\sum_{a \in A} \text{frequency}(a, t)}{|R_t|}$$
*Note:* This separate metric **may exceed 1.0 (100%)** when reviews describe co-occurring defects across multiple subsystems.

### D. Average Complaint Severity
Each review $r$ receives a text severity $s_{\text{text}}(r) \in [0, 1]$ from the sentiment model and an optional star rating $y(r) \in [1, 5]$. Combined severity is:
$$s_{\text{comb}}(r) = 0.7 \cdot s_{\text{text}}(r) + 0.3 \cdot \left(\frac{5 - y(r)}{4}\right)$$
For period $t$:
$$\bar{s}_t = \frac{1}{|R_t^{\text{complaint}}|} \sum_{r \in R_t^{\text{complaint}}} s_{\text{comb}}(r)$$

### E. Safe Complaint Growth Rate
Change in complaint proportion compared with the previous period, with a minimum baseline floor ($0.05$) to guard against zero-denominator explosion:
$$g_t = \frac{p_t - p_{t-1}}{\max(p_{t-1}, 0.05)}$$

### F. Symptom Diversity
Ratio of unique active complaint categories to total supported aspects ($|A| = 9$):
$$d_t = \frac{|\{a \in A : \text{frequency}(a, t) > 0\}|}{|A|}$$

### G. Review Arrival Velocity & Ratio
Let $v_t = |R_t|$ be review volume. Its historical exponentially weighted moving average baseline is:
$$\text{EWMA}(v)_t = \alpha v_{t-1} + (1-\alpha) \text{EWMA}(v)_{t-1}, \quad \alpha = \frac{2}{\text{span} + 1}$$
The velocity ratio is:
$$\rho_t = \frac{v_t}{\text{EWMA}(v)_t + \epsilon}$$

### F. Product Risk Score Formula
Signals are normalized onto $[0, 1]$:
* $\hat{p}_t = \text{clip}(p_t, 0, 1)$
* $\hat{s}_t = \text{clip}(\bar{s}_t, 0, 1)$
* $\hat{g}_t = \max(0, \tanh(g_t))$
* $\hat{d}_t = \text{clip}(d_t, 0, 1)$
* $\hat{v}_t = \tanh(\max(0, \rho_t - 1))$

The composite risk score is:
$$\text{RiskScore}_t = 100 \times \left( w_1 \hat{p}_t + w_2 \hat{s}_t + w_3 \hat{g}_t + w_4 \hat{d}_t + w_5 \hat{v}_t \right)$$

**Default Engineering Weights:**
* Frequency ($w_1$): $0.25$ (25%)
* Severity ($w_2$): $0.20$ (20%)
* Growth ($w_3$): $0.25$ (25%)
* Diversity ($w_4$): $0.15$ (15%)
* Velocity ($w_5$): $0.15$ (15%)

**Risk Bands:**
* **Low**: $0 \le \text{Score} < 25$
* **Moderate**: $25 \le \text{Score} < 50$
* **High**: $50 \le \text{Score} < 75$
* **Critical**: $75 \le \text{Score} \le 100$

### G. Anomaly Detection (No Future Leakage)
For period $t > \text{min\_history}$, baselines $\mu$ and $\sigma$ are calculated strictly over periods $\{1, \dots, t-1\}$:
$$z_t = \frac{x_t - \mu_{1:t-1}}{\sigma_{1:t-1}}$$
If $z_t > \theta$ (default $\theta = 2.0$), period $t$ is flagged with an evidence breakdown.

---

## 6. Technology Stack

* **Language**: Python 3.8+ (tested on Python 3.12)
* **Data Processing**: Pandas, NumPy
* **NLP & Sentiment**: HuggingFace Transformers (`distilbert-base-uncased-finetuned-sst-2-english`), PyTorch, rule-based VADER-style fallback
* **Machine Learning & Metrics**: Scikit-Learn
* **Visualization & UI**: Streamlit, Plotly Express & Graph Objects
* **Testing**: Pytest

---

## 7. Dataset Schema & Preparation Guide

### Expected CSV Schema

| Column Name | Required | Type | Description | Accepted Aliases |
|-------------|----------|------|-------------|------------------|
| `product_id` | **Yes** | String | Unique product identifier | `product`, `asin`, `item_id`, `sku` |
| `review_date` | **Yes** | Date/String | Timestamp of review | `date`, `timestamp`, `review_time`, `created_at` |
| `review_text` | **Yes** | String | Customer review content | `text`, `review`, `body`, `comment`, `content` |
| `rating` | Optional | Float/Int (1–5)| Star score | `score`, `overall`, `stars`, `star_rating` |

### Preparing Public Datasets (e.g., Amazon Reviews)
1. Download a category dataset (e.g., Electronics or Cell Phones & Accessories).
2. Ensure columns are mapped:
   - `asin` $\rightarrow$ `product_id`
   - `unixReviewTime` or `reviewTime` $\rightarrow$ `review_date`
   - `reviewText` $\rightarrow$ `review_text`
   - `overall` $\rightarrow$ `rating`
3. Export to a standard UTF-8 encoded `.csv`. The loader will automatically map these aliases.

---

## 8. Installation & Setup Guide (Windows PowerShell)

### Step 1: Clone or Open Project Folder
```powershell
cd "c:\Users\PRANAV REDDY\Downloads\ISP BUILD PROJECT"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Generate Synthetic Demonstration Data
```powershell
python scripts/generate_demo_data.py
```

### Step 5: Run the Pytest Test Suite
```powershell
pytest tests/ -v
```

### Step 6: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Dashboard Walkthrough

The Streamlit dashboard contains 7 specialized pages:

1. **📋 Overview**: Problem statement, research question, architecture summary, and dataset high-level status.
2. **📂 Data Explorer**: CSV file upload with interactive column mapping, synthetic demonstration data loader, validation report, and analysis pipeline execution trigger.
3. **🔎 Complaint Analysis**: Aspect distributions, sentiment label pie charts, severity histograms, and aspect-filtered raw review inspector.
4. **📈 Temporal Trends**: Interactive dual-axis charts of complaint proportion vs review velocity, per-aspect trend comparisons, growth rates, and symptom diversity over time.
5. **⚠️ Risk Monitoring**: Current product risk gauge, risk band indicator, breakdown of 5 signal contributions, historical risk score trajectory with anomaly markers, and explainable evidence alerts.
6. **📊 Model Evaluation**: Dynamic slider comparison between the frequency baseline and multi-signal model, agreement metrics, ablation study table and plots, and academic evaluation status.
7. **📖 Methodology**: Formal mathematical definitions, signal normalization equations, risk weighting rationale, and limitation disclosures.

---

## 10. Evaluation Methodology & Academic Disclaimers

### Baseline Comparison
A simple **Frequency Baseline** flags any period where overall complaint proportion exceeds a static threshold (default $30\%$). The proposed **Multi-Signal Model** evaluates the co-evolution of severity, growth, diversity, and velocity alongside frequency.

### Ablation Study
The system conducts an ablation study by systematically removing each signal ($w_i = 0$, renormalizing remaining weights) to isolate individual signal influence on peak and mean risk ratings.

### Critical Academic Disclaimers
* **Synthetic vs Real Data**: Synthetic demo records are generated strictly to verify software mechanics and should never be reported as real-world customer findings.
* **Absence of Ground-Truth Failure Labels**: Standard classification metrics (Precision, Recall, F1) require independently verified product recall or warranty RMA failure dates. Without verified failure labels, the system presents descriptive anomaly detection and agreement metrics.
* **Heuristic Risk vs Physical Failure**: The risk score is an empirical early warning indicator reflecting customer complaints—it does not constitute physical proof of hardware failure or an exact failure date forecast.

---

## 11. Known Limitations & Future Work

* **Aspect Lexicon**: Keyword matching can be sensitive to novel slang or complex syntax. Future work can incorporate domain-adapted Sentence-BERT embeddings.
* **Weight Calibration**: Default weights are heuristic engineering choices. Future work with labeled failure datasets could optimize weights via Bayesian calibration or logistic regression.
* **Review Verification**: Reviews are treated equally regardless of "Verified Purchase" badges or reviewer credibility.

---

## 12. Git & GitHub Setup Instructions

To push this codebase to a GitHub repository:
```powershell
git init
git add .
git commit -m "Initial commit: Early Product Failure Prediction System"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
