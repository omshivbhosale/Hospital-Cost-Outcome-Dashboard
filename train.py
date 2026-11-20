"""Predict a hospital's Excess Readmission Ratio from cost, patient-mix and case-mix data.

Predicted and Expected Readmission Rate are deliberately NOT used: CMS defines
Excess Readmission Ratio = Predicted / Expected, so they would leak the target.
The Number of Readmissions column and the hospital star rating (which includes a
readmission group) are excluded for the same reason.

Run from the repository root:  python train.py
Writes: results/metrics.json, results/feature_importance.csv,
        IMSE_586_Project_Code/best_model.pkl, data/processed_data.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 586
ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

TARGET = "Excess Readmission Ratio"
MONEY_COLS = ["Tot_Pymt_Amt", "Tot_Mdcr_Pymt_Amt", "Tot_Submtd_Cvrd_Chrg"]
# Original feature set (totals, which mostly track hospital size)
NUMERIC = MONEY_COLS + ["Cost_Ratio_Total_vs_Medicare", "Cost_Ratio_Charges_vs_Medicare"]
CATEGORICAL = ["Hospital Type", "Hospital Ownership", "State"]


def ccn(series):
    return series.astype(str).str.strip().str.zfill(6)


def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce")


def load():
    """Merge the three CMS files into one row per hospital with all candidate features."""
    hrrp = pd.read_csv(DATA / "Hospital_Readmissions_Reduction_Program_Hospital.csv")
    gen = pd.read_csv(DATA / "Hospital_General_Information.csv")
    mdcr = pd.read_csv(DATA / "Medicare_IP_Hospitals_by_Provider_2023.csv", encoding="latin1")
    mdcr.columns = [c.replace("ï»¿", "") for c in mdcr.columns]  # stray BOM bytes
    mdcr = mdcr.rename(columns={"Rndrng_Prvdr_CCN": "Facility ID"})
    for df in (hrrp, gen, mdcr):
        df["Facility ID"] = ccn(df["Facility ID"])

    # Medicare file: numeric columns come in as text with thousands separators
    for c in mdcr.columns:
        if c.startswith(("Tot_", "Bene_")):
            mdcr[c] = to_num(mdcr[c])

    # HRRP: target (mean over the six conditions) and case mix by discharge volume
    hrrp[TARGET] = pd.to_numeric(hrrp[TARGET], errors="coerce")
    hrrp["Number of Discharges"] = pd.to_numeric(hrrp["Number of Discharges"], errors="coerce")
    target = hrrp.groupby("Facility ID")[TARGET].mean()
    vol = hrrp.pivot_table(index="Facility ID", columns="Measure Name",
                           values="Number of Discharges", aggfunc="sum")
    mix = vol.div(vol.sum(axis=1), axis=0).add_prefix("mix_")
    mix["hrrp_log_volume"] = np.log1p(vol.sum(axis=1))

    df = gen.merge(mdcr, on="Facility ID").drop_duplicates(subset="Facility ID")
    df = df.set_index("Facility ID").join(target, how="inner").join(mix)

    # Cost features: totals plus per-discharge versions
    df["Cost_Ratio_Total_vs_Medicare"] = df["Tot_Pymt_Amt"] / df["Tot_Mdcr_Pymt_Amt"]
    df["Cost_Ratio_Charges_vs_Medicare"] = df["Tot_Submtd_Cvrd_Chrg"] / df["Tot_Mdcr_Pymt_Amt"]
    d = df["Tot_Dschrgs"]
    df["pay_per_discharge"] = df["Tot_Pymt_Amt"] / d
    df["charge_per_discharge"] = df["Tot_Submtd_Cvrd_Chrg"] / d
    df["mdcr_pay_per_discharge"] = df["Tot_Mdcr_Pymt_Amt"] / d
    df["avg_length_of_stay"] = df["Tot_Days"] / d
    df["log_discharges"] = np.log1p(d)

    # Patient mix of the Medicare patients each hospital treats
    n = df["Tot_Benes"]
    df["avg_age"] = df["Bene_Avg_Age"]
    df["avg_risk_score"] = df["Bene_Avg_Risk_Scre"]
    df["pct_dual_eligible"] = df["Bene_Dual_Cnt"] / n
    df["pct_age_84_plus"] = df["Bene_Age_GT_84_Cnt"] / n
    df["pct_age_under_65"] = df["Bene_Age_LT_65_Cnt"] / n
    df["pct_black"] = df["Bene_Race_Black_Cnt"] / n
    df["pct_female"] = df["Bene_Feml_Cnt"] / n
    df["rural_urban_code"] = to_num(df["Rndrng_Prvdr_RUCA"])
    return df.replace([np.inf, -np.inf], np.nan)


def feature_lists(df):
    chronic = [c for c in df.columns if c.startswith("Bene_CC_")]
    per_discharge = ["pay_per_discharge", "charge_per_discharge", "mdcr_pay_per_discharge",
                     "avg_length_of_stay", "log_discharges"]
    patient = ["avg_age", "avg_risk_score", "pct_dual_eligible", "pct_age_84_plus",
               "pct_age_under_65", "pct_black", "pct_female", "rural_urban_code"] + chronic
    case_mix = [c for c in df.columns if c.startswith("mix_")] + ["hrrp_log_volume"]
    return NUMERIC, NUMERIC + per_discharge + patient + case_mix


def scores(y, pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred))}


def main():
    df = load()
    df.reset_index().to_csv(DATA / "processed_data.csv", index=False)
    base_num, full_num = feature_lists(df)

    df = df.dropna(subset=[TARGET] + MONEY_COLS)
    cats = pd.get_dummies(df[["Hospital Type", "Hospital Ownership"]], dtype=float)
    X_base = df[base_num].join(cats)
    X_full = df[full_num].join(cats)
    y = df[TARGET]

    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=SEED)
    cv = KFold(5, shuffle=True, random_state=SEED)
    results = {"n_hospitals": int(len(df)), "n_features_original": int(X_base.shape[1]),
              "n_features_final": int(X_full.shape[1])}

    results["mean_baseline"] = scores(y[idx_test], np.full(len(idx_test), y[idx_train].mean()))

    def ridge():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-1, 4, 20)))

    # Original cost-only features, for comparison
    m = ridge().fit(X_base.loc[idx_train], y[idx_train])
    results["ridge_original_features"] = scores(y[idx_test], m.predict(X_base.loc[idx_test]))

    # Final feature set
    m = ridge().fit(X_full.loc[idx_train], y[idx_train])
    results["ridge_final_features"] = scores(y[idx_test], m.predict(X_full.loc[idx_test]))

    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                             random_state=SEED, n_jobs=-1))
    rf_cv = cross_val_score(rf, X_full.loc[idx_train], y[idx_train], cv=cv, scoring="r2", n_jobs=1)
    rf.fit(X_full.loc[idx_train], y[idx_train])
    results["random_forest_final_features"] = {
        "cv_r2_train": float(rf_cv.mean()), **scores(y[idx_test], rf.predict(X_full.loc[idx_test]))}

    grid = GridSearchCV(
        HistGradientBoostingRegressor(random_state=SEED),
        {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1],
         "max_iter": [200, 400], "min_samples_leaf": [20, 40]},
        cv=cv, scoring="r2", n_jobs=-1,
    )
    grid.fit(X_full.loc[idx_train], y[idx_train])
    best = grid.best_estimator_
    results["gradient_boosting_final_features"] = {
        "best_params": grid.best_params_,
        "cv_r2_train": float(grid.best_score_),
        **scores(y[idx_test], best.predict(X_full.loc[idx_test])),
    }

    # Pick the final model by cross-validation on the training hospitals, not by the test set
    candidates = {"random_forest": rf, "gradient_boosting": best}
    cv_scores = {"random_forest": results["random_forest_final_features"]["cv_r2_train"],
                 "gradient_boosting": results["gradient_boosting_final_features"]["cv_r2_train"]}
    chosen = max(cv_scores, key=cv_scores.get)
    best = candidates[chosen]
    results["final_model"] = {"name": chosen, **scores(y[idx_test], best.predict(X_full.loc[idx_test]))}

    # Which features matter, measured on the held-out hospitals
    pi = permutation_importance(best, X_full.loc[idx_test], y[idx_test], scoring="r2",
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_full.columns, "importance": pi.importances_mean}
                       ).sort_values("importance", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    joblib.dump({"model": best, "features": list(X_full.columns)},
                ROOT / "IMSE_586_Project_Code" / "best_model.pkl", compress=3)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(imp.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
