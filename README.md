# Hospital Cost & Outcome Dashboard

> Visual analytics and predictive modeling on federal CMS healthcare data: what explains which hospitals are penalized for excess readmissions?

**Stack:** Python · pandas · scikit-learn · Random Forest · Plotly Dash · Matplotlib/Seaborn
**Domain:** Healthcare · Big Data · Predictive modeling · Visual analytics

---

## Problem
Hospital readmissions are a major cost and quality signal, and CMS penalizes facilities with excess readmissions. The goal: integrate disparate federal datasets and test which cost, patient-mix and case-mix factors predict a hospital's **Excess Readmission Ratio**.

## Data
Three public **CMS** datasets (in `data/`) joined into one hospital-level table:
- **HRRP** (Hospital Readmissions Reduction Program)
- **Medicare Inpatient Providers** (2023)
- **Hospital General Information**

Joined on the **6-digit CMS Certification Number (CCN)**, converted to a zero-padded string so the key matches across sources. HRRP was averaged to one row per hospital. After joining and dropping hospitals with missing cost fields, 2,842 hospitals remain for modeling.

## Approach
- **Data engineering:** ingestion, cleaning of monetary fields, inner-join merge on CCN, one row per hospital (2,842 hospitals after dropping missing cost fields).
- **Original features (v1):** total payments, Medicare payments, covered charges, two cost ratios, hospital type, ownership and state. These mostly track hospital size.
- **Added features (final):**
  - *Per-discharge economics:* payment, charge and Medicare payment per discharge, average length of stay, discharge volume.
  - *Patient mix* (Medicare file): average age, average risk score, share dual-eligible, share over 84 and under 65, share female and Black, rural/urban code, and chronic-condition rates (COPD, diabetes, dementia, heart failure and others).
  - *Case mix* (HRRP file): each hospital's share of discharges across the six HRRP conditions, plus total HRRP volume.
- **Models:** Ridge regression, Random Forest and tuned gradient boosting, compared with a mean-only baseline on a held-out 20% test set. The final model is chosen by 5-fold cross-validation on the training hospitals, not by the test set.
- **Dashboard:** Plotly Dash app with filters by region, state, hospital type, ownership and quality score.

### What is excluded, and why
CMS defines the Excess Readmission Ratio as **Predicted Readmission Rate ÷ Expected Readmission Rate**. Using either rate as an input lets a model rebuild the target by arithmetic, which inflates R² without saying anything about cost drivers. Both are excluded. So are the Number of Readmissions column and the hospital star rating, because both are built from readmission outcomes.

## Results

| Model (20% held-out test set) | RMSE | R² |
|---|---|---|
| Mean baseline | 0.0557 | -0.002 |
| Ridge, original cost features | 0.0548 | 0.032 |
| Ridge, final features | 0.0483 | 0.248 |
| Random Forest, final features | 0.0428 | 0.408 |
| Gradient boosting (tuned), final features | 0.0460 | 0.319 |

- **Cost features alone predict almost nothing** (R² about 0.03): the first version of this project looked far better only because it used the Predicted and Expected rates.
- **Patient mix and case mix carry the signal.** Adding them takes the test R² from about 0.03 to 0.41 for the Random Forest, the model selected by cross-validation.
- **Treat 0.41 as an upper estimate.** Cross-validated R² on the training hospitals is lower (Random Forest 0.26, gradient boosting 0.24), and the test split is only about 570 hospitals. A fair summary is "roughly 0.25 to 0.40 of the variation".
- **Most important features** (permutation importance on the test set): total HRRP discharge volume, the share of COPD and hip/knee discharges, average patient risk score, and dementia and COPD prevalence. See `results/feature_importance.csv`.

The relationship is associational: these features describe who a hospital treats and how much it treats, not what causes readmissions.

## Run it

```bash
pip install -r requirements.txt
python train.py                              # rebuilds features, trains, writes results/ and the model
cd IMSE_586_Project_Code
jupyter nbconvert --to notebook --execute main_notebook.ipynb   # or open it in Jupyter
```

To launch the dashboard, run `train.py` first (it writes `data/processed_data.csv`), then open `IMSE_586_Project_Code/dashoard.ipynb` and run all cells.

## My role
Team of 3 (IMSE 586 Big Data Analytics & Visualization). Contributed to data ingestion/merging, feature engineering, modeling, and the dashboard.
