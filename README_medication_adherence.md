# 💊 Medication Adherence Prediction Pipeline

> End-to-end production ML pipeline predicting prescription non-adherence — XGBoost + LightGBM ensemble with SHAP explainability, temporal feature engineering, and drift detection.

**Author:** [Naveed Abbas Themaliparambil](https://github.com/Naveed-0730) · [LinkedIn](https://linkedin.com/in/naveedabbasds)

---

## Overview

Only 50% of patients with chronic conditions adhere to prescribed medication. Early identification of at-risk patients enables proactive intervention — reducing discontinuation, improving outcomes, and lowering healthcare costs.

This pipeline takes raw prescription refill data through to production-ready patient risk scores with clinical intervention recommendations.

---

## Pipeline Architecture

```
Raw Refill Data
      │
      ▼
Temporal Feature Engineering (35 features — 14 temporal)
      │
      ▼
XGBoost + LightGBM Ensemble Training
      │
      ▼
SHAP Explainability (global + local)
      │
      ▼
Model Evaluation (ROC-AUC, Precision-Recall, Risk Stratification)
      │
      ▼
PSI + KS Drift Detection
      │
      ▼
Patient Risk Scoring → Intervention Recommendations
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 50,000 patients |
| Features engineered | 35 (inc. 14 temporal) |
| Ensemble ROC-AUC | 0.87+ |
| Models | XGBoost + LightGBM |
| Explainability | SHAP (global + local) |
| Monitoring | PSI + KS drift detection |

---

## Feature Engineering

Key temporal features engineered from raw refill data:

| Feature | Description |
|---------|-------------|
| `refill_regularity_score` | Inverse of refill gap standard deviation — measures consistency |
| `pdc_score` | Proportion of Days Covered — clinical adherence threshold |
| `pdc_risk_band` | 4-band risk classification based on PDC |
| `days_overdue` | Days since refill was expected |
| `missed_refill_rate` | Monthly missed refill rate (3-month window) |
| `risk_score_composite` | Weighted composite of top 3 risk signals |
| `late_to_early_ratio` | Ratio of late vs early refills |
| `delivery_failure_rate` | Delivery failures per total refills |

---

## Tech Stack

```
Python · XGBoost · LightGBM · SHAP · Scikit-learn
Pandas · NumPy · Matplotlib · Seaborn · SciPy
```

---

## How to Run

**Option 1 — Google Colab (recommended, no setup)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naveed-0730/medication-adherence-ml/blob/main/pharmacy2u_adherence_project.ipynb)

**Option 2 — Local**

```bash
git clone https://github.com/Naveed-0730/medication-adherence-ml
cd medication-adherence-ml
pip install -r requirements.txt
jupyter notebook pharmacy2u_adherence_project.ipynb
```

---

## Output — Patient Risk Scoring

The pipeline outputs risk scores and intervention recommendations for each patient:

| Risk Band | Score | Recommended Action |
|-----------|-------|--------------------|
| CRITICAL 🔴 | 75–100 | Immediate pharmacist outreach + GP alert |
| HIGH 🟠 | 55–74 | Proactive SMS reminder + adherence support |
| MEDIUM 🟡 | 35–54 | Automated reminder sequence |
| LOW 🟢 | 20–34 | Standard monitoring |
| VERY LOW ✅ | 0–19 | No action required |

---

## Related Projects

- [Credit Risk ML Pipeline](https://github.com/Naveed-0730) — WoE scorecard, PD/LGD/EAD on 2.2M records
- [Portfolio](https://naveed-0730.github.io/naveed-portfolio) — Full project showcase

---

## About

MSc Data Science and Analytics · University of Leeds · First Class  
Full UK right to work · Available immediately  
Enhanced DBS Certified

[naveedabbas0872@gmail.com](mailto:naveedabbas0872@gmail.com) · [linkedin.com/in/naveedabbasds](https://linkedin.com/in/naveedabbasds)
