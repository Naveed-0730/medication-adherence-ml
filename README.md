# 💊 Medication Adherence Prediction Pipeline

> End-to-end production ML pipeline predicting prescription non-adherence — XGBoost + LightGBM ensemble, ROC-AUC 0.88, SHAP explainability, temporal feature engineering, and PSI/KS drift detection.

**Author:** [Naveed Abbas Themaliparambil](https://github.com/Naveed-0730) · [LinkedIn](https://linkedin.com/in/naveedabbasds) · [Portfolio](https://naveed-0730.github.io/naveed-portfolio)

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
| XGBoost ROC-AUC | 0.883 |
| LightGBM ROC-AUC | 0.857 |
| **Ensemble ROC-AUC** | **0.879** |
| Explainability | SHAP (global + local) |
| Monitoring | PSI + KS drift detection |

---

## Results

### Model Performance

![Model Performance Dashboard](model_performance.png)

**ROC curves** show all three models well above the random baseline. The ensemble (AUC 0.879) outperforms both individual models by combining their strengths. The **risk stratification chart** demonstrates strong separation — the Very High risk band has an 11% non-adherence rate vs 0% in the Very Low band, confirming the model meaningfully identifies at-risk patients.

---

### SHAP Feature Importance

![SHAP Feature Importance](shap_feature_importance.png)

`pdc_score` (Proportion of Days Covered) is the single strongest predictor — consistent with NHS clinical evidence. `delivery_failure_rate` and `missed_refills_3m` are the next most important, both directly reflecting patient engagement with their prescription routine.

---

### SHAP Beeswarm — Feature Impact Direction

![SHAP Beeswarm](shap_beeswarm.png)

- **pdc_score**: High values (blue) reduce risk; low values (red) strongly increase it — exactly as expected clinically
- **delivery_failure_rate**: High values push risk up — patients with repeated delivery failures are disengaging
- **missed_refills_3m**: High missed refill counts consistently increase non-adherence probability
- **app_logins_30d**: High engagement (blue) reduces risk — digital engagement is a protective signal

---

### Drift Detection Report

```
🔍 DRIFT DETECTION REPORT
============================================================
              feature  ks_statistic  ks_pvalue    psi  drift_detected
            pdc_score        0.2584     0.0000 0.7592            True
days_since_last_refill        0.3265     0.0000 0.5947            True
       num_refills_12m        0.0104     0.3661 0.0020           False
   avg_refill_gap_days        0.0072     0.8113 0.0011           False
  risk_score_composite        0.0084     0.6436 0.0007           False
     missed_refills_3m        0.4283     0.0000 0.0001            True

⚠️  ALERT: 3 features show significant drift — retraining recommended
```

The PSI/KS monitoring correctly detected distribution shift in `pdc_score`, `days_since_last_refill`, and `missed_refills_3m` — the three features that were deliberately modified to simulate a real-world scenario where patients become less adherent over time. Stable features (`num_refills_12m`, `avg_refill_gap_days`) correctly showed no drift. In production, this alert would trigger an automated retraining job before model performance degrades.

---

### Patient Risk Scoring — Production Output

```
📊 PATIENT RISK SCORING — PRODUCTION OUTPUT
================================================================================
patient_id  risk_score  risk_band  recommended_intervention
  P036958        69.5       HIGH   Proactive SMS reminder + adherence support
  P042724        47.6     MEDIUM   Automated reminder sequence
  P010822        30.0        LOW   Standard monitoring
  P033553        20.8        LOW   Standard monitoring
  P000199        16.9   VERY LOW   Standard monitoring
  P039489        13.1   VERY LOW   Standard monitoring
  P004144        10.3   VERY LOW   Standard monitoring
  P049498        10.2   VERY LOW   Standard monitoring
  P012447        10.0   VERY LOW   Standard monitoring
  P009427         9.9   VERY LOW   Standard monitoring
```

Each patient receives a 0–100 risk score and a prioritised intervention recommendation. P036958 (score 69.5) is flagged HIGH — this patient would receive a proactive pharmacist SMS and adherence support call. The distribution reflects realistic population prevalence — most patients are low risk, with a small subset requiring proactive intervention.

---

## Feature Engineering

Key temporal features engineered from raw refill data:

| Feature | Description |
|---------|-------------|
| `pdc_score` | Proportion of Days Covered — standard NHS clinical adherence measure |
| `pdc_risk_band` | 4-band risk classification (below 0.8 = clinically non-adherent) |
| `refill_regularity_score` | Inverse of refill gap standard deviation — measures consistency |
| `days_overdue` | Days since refill was expected |
| `missed_refill_rate` | Monthly missed refill rate (3-month window) |
| `risk_score_composite` | Weighted composite of top 3 risk signals |
| `late_to_early_ratio` | Ratio of late vs early refills |
| `delivery_failure_rate` | Delivery failures per total refills |

---

## Tech Stack

```
Python · XGBoost · LightGBM · SHAP · Scikit-learn
Pandas · NumPy · Matplotlib · Seaborn · SciPy · MLflow
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

| Risk Band | Score | Recommended Action |
|-----------|-------|--------------------|
| CRITICAL 🔴 | 75–100 | Immediate pharmacist outreach + GP alert |
| HIGH 🟠 | 55–74 | Proactive SMS reminder + adherence support |
| MEDIUM 🟡 | 35–54 | Automated reminder sequence |
| LOW 🟢 | 20–34 | Standard monitoring |
| VERY LOW ✅ | 0–19 | No action required |

---

## About

MSc Data Science and Analytics · University of Leeds · First Class  
Full UK right to work · Available immediately · Enhanced DBS Certified

[naveedabbas0872@gmail.com](mailto:naveedabbas0872@gmail.com) · [linkedin.com/in/naveedabbasds](https://linkedin.com/in/naveedabbasds)
