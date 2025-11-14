# training/train_models.py
"""
Train one model per dataset and save (preprocessor + model) into ../models/.
Adjust nrows for CICIDS2017 if memory is limited.
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return pre

def fit_and_save(X, y, model, tag):
    pre = build_preprocessor(X)
    print(f"[{tag}] fitting preprocessor on {X.shape} ...")
    pre.fit(X)
    Xp = pre.transform(X)
    if len(np.unique(y)) > 1:
        print(f"[{tag}] balancing with SMOTE ...")
        sm = SMOTE(random_state=42)
        Xr, yr = sm.fit_resample(Xp, y)
    else:
        Xr, yr = Xp, y
    print(f"[{tag}] training model ...")
    model.fit(Xr, yr)
    joblib.dump(pre, MODELS_DIR / f"pre_{tag}.pkl")
    joblib.dump(model, MODELS_DIR / f"model_{tag}.pkl")
    print(f"[{tag}] saved pre_{tag}.pkl and model_{tag}.pkl")

# ---------------- TRAIN NSL-KDD (XGBoost) ----------------
nsl_path = DATA_DIR / "KDDTest_converted.csv"
if nsl_path.exists():
    df = pd.read_csv(nsl_path)
    # detect label col
    label = next((c for c in ['label','class','Label','attack_cat','target'] if c in df.columns), df.columns[-1])
    y = df[label].apply(lambda x: 0 if str(x).lower() in ['normal','0','benign'] else 1).astype(int)
    X = df.drop(columns=[label])
    xgb = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    fit_and_save(X, y, xgb, "NSL")
else:
    print("KDDTest_converted.csv not found in data/ — skipping NSL training")

# ---------------- TRAIN UNSW-NB15 (ExtraTrees) ----------------
unsw_path = DATA_DIR / "UNSW_NB15_testing-set_converted.csv"
if unsw_path.exists():
    df = pd.read_csv(unsw_path)
    label = next((c for c in ['label','Label','attack_cat','class','target'] if c in df.columns), df.columns[-1])
    y = df[label].apply(lambda x: 0 if str(x).lower() in ['normal','0','benign'] else 1).astype(int)
    X = df.drop(columns=[label])
    et = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    fit_and_save(X, y, et, "UNSW")
else:
    print("UNSW file not found in data/ — skipping UNSW training")

# ---------------- TRAIN CICIDS2017 (RandomForest) using SAMPLE ----------------
cic_path = DATA_DIR / "CICIDS2017.csv"
if cic_path.exists():
    print("CICIDS2017 found — will train on a SAMPLE (reduce nrows if needed).")
    nrows = 200000  # change lower if RAM is limited
    df = pd.read_csv(cic_path, nrows=nrows)
    label = next((c for c in ['Label','label','Label_name','class','BENIGN'] if c in df.columns), df.columns[-1])
    y = df[label].apply(lambda x: 0 if str(x).lower() in ['normal','benign','0'] else 1).astype(int)
    X = df.drop(columns=[label])
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    fit_and_save(X, y, rf, "CICIDS")
else:
    print("CICIDS2017.csv not found in data/ — skipping CICIDS training")

print("Finished training script.")
