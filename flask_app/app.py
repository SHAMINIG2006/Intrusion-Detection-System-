# flask_app/app.py
import os
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

app = Flask(__name__)
app.secret_key = "replace-me-with-secure-key"

# load models that exist
loaded = {}
for prefile in MODELS_DIR.glob("pre_*.pkl"):
    tag = prefile.stem.replace("pre_","")
    try:
        pre = joblib.load(MODELS_DIR / f"pre_{tag}.pkl")
        model = joblib.load(MODELS_DIR / f"model_{tag}.pkl")
        loaded[tag] = {"pre": pre, "model": model}
    except Exception as e:
        print(f"Could not load both pre & model for {tag}: {e}")

print("Loaded models:", list(loaded.keys()))

def detect_dataset(df: pd.DataFrame):
    cols = set([c.lower() for c in df.columns])
    # NSL signature
    if 'protocol_type' in cols or ('service' in cols and 'flag' in cols):
        return "NSL"
    # UNSW signature
    if 'proto' in cols and 'sbytes' in cols:
        return "UNSW"
    # CICIDS heuristics (many fwd/bwd/flow features)
    if any('fwd' in c.lower() or 'bwd' in c.lower() or 'flow' in c.lower() for c in df.columns):
        return "CICIDS"
    return None

def prepare_for_predict(pre, df):
    # ensure expected columns exist; fill missing with zero
    try:
        if hasattr(pre, "feature_names_in_") and pre.feature_names_in_ is not None:
            exp = list(pre.feature_names_in_)
            df2 = df.copy()
            for c in exp:
                if c not in df2.columns:
                    df2[c] = 0
            return df2[exp]
        else:
            return df.select_dtypes(include=[np.number])
    except Exception:
        return df.select_dtypes(include=[np.number])

@app.route("/")
def home():
    return render_template("home.html", models=list(loaded.keys()))

@app.route("/submit", methods=["GET","POST"])
def submit():
    if request.method == "POST":
        # If CSV uploaded
        f = request.files.get("csvfile")
        if f and f.filename:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                flash("Could not read uploaded CSV.")
                return redirect(url_for("submit"))
            ds = detect_dataset(df)
            if ds is None:
                flash("Could not auto-detect dataset. Ensure your CSV is formatted like NSL, UNSW or CICIDS.")
                return redirect(url_for("submit"))
            # save temp
            tmp = ROOT / "tmp_upload.csv"
            df.to_csv(tmp, index=False)
            preview_html = df.head(8).to_html(classes="table table-sm table-striped")
            return render_template("predict.html", detected=ds, preview=preview_html, filename=str(tmp.name))
        else:
            # manual input
            dataset = request.form.get("dataset")
            # collect fields
            features = {}
            for k, v in request.form.items():
                if k == "dataset" or v.strip() == "":
                    continue
                # try convert to numeric
                try:
                    if "." in v:
                        features[k] = float(v)
                    else:
                        features[k] = int(v)
                except:
                    features[k] = v
            if not dataset:
                flash("Select dataset and provide inputs.")
                return redirect(url_for("submit"))
            df = pd.DataFrame([features])
            return do_prediction_from_df(df, dataset)
    return render_template("submit.html", models=list(loaded.keys()))

def do_prediction_from_df(df, dataset_tag=None, filename=None):
    # dataset_tag like "NSL", "UNSW", "CICIDS"
    ds = dataset_tag if dataset_tag else detect_dataset(df)
    if ds is None:
        return "Could not detect dataset type", 400
    tag = ds  # maps directly to keys used in training script
    if tag not in loaded:
        return f"No trained model for dataset {tag}. Please train and provide model files.", 400
    pre = loaded[tag]["pre"]
    model = loaded[tag]["model"]
    Xp = prepare_for_predict(pre, df)
    X_t = pre.transform(Xp) if hasattr(pre, "transform") else Xp.values
    # align feature count to model if necessary
    if hasattr(model, "n_features_in_") and X_t.shape[1] != model.n_features_in_:
        diff = model.n_features_in_ - X_t.shape[1]
        if diff > 0:
            X_t = np.pad(X_t, ((0,0),(0,diff)), mode="constant", constant_values=0)
        else:
            X_t = X_t[:, :model.n_features_in_]
    preds = model.predict(X_t)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_t)[:,1]  # probability of class 1
    results = []
    for i, p in enumerate(preds):
        results.append({"index": i, "prediction": int(p), "prob": float(probs[i]) if probs is not None else None})
    # If filename provided (uploaded csv), include sample preview
    preview = None
    if filename:
        preview = pd.read_csv(ROOT / filename).head(6).to_html(classes="table table-sm table-striped")
    return render_template("predict.html", detected=ds, results=results, preview=preview)

@app.route("/predict_file", methods=["POST"])
def predict_file():
    # read tmp_upload.csv saved by submit
    path = ROOT / "tmp_upload.csv"
    if not path.exists():
        flash("No uploaded CSV found.")
        return redirect(url_for("submit"))
    df = pd.read_csv(path)
    ds = detect_dataset(df)
    return do_prediction_from_df(df, dataset_tag=ds, filename=path.name)

if __name__ == "__main__":
    app.run(debug=True)
