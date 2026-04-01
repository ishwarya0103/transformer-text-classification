"""
Boilerplate submission pipeline for SPR 2026 Mammography BI-RADS Classification.

Competition: predict BI-RADS category (0-6) from mammography report text.
Metric: macro F1-score.

Usage:
    1. Implement your model in the `build_model` function.
    2. Run this script: python pipeline.py
    3. Output: submission.csv ready for upload.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(".")
SEED = 42
N_FOLDS = 5
TARGET_COL = "target"
TEXT_COL = "report"
BIRADS_CLASSES = list(range(7))  # 0, 1, 2, 3, 4, 5, 6

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "submission.csv")
    return train, test, submission


# ---------------------------------------------------------------------------
# Preprocessing — customize as needed
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Clean / normalize a single report string."""
    if pd.isna(text):
        return ""
    text = text.strip()
    # TODO: add your preprocessing (lowercase, remove dates, etc.)
    return text


# ---------------------------------------------------------------------------
# Model — PLUG YOUR MODEL HERE
# ---------------------------------------------------------------------------
def build_model():
    """
    Return an object that implements .fit(X, y) and .predict(X).

    X is a pd.Series of report strings, y is the target array.

    Examples you might swap in:
        - TF-IDF + LogisticRegression / LightGBM
        - Sentence-transformer embeddings + classifier
        - Fine-tuned multilingual LLM (e.g. XLM-R, BERTimbau)
        - Zero-shot / few-shot LLM prompting wrapper
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate(train: pd.DataFrame):
    """Run stratified K-fold CV and report macro-F1 per fold."""
    X = train[TEXT_COL].apply(preprocess)
    y = train[TARGET_COL].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    oof_preds = np.full(len(train), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        score = f1_score(y_val, preds, average="macro")
        fold_scores.append(score)
        print(f"  Fold {fold}: macro-F1 = {score:.4f}")

    overall = f1_score(y, oof_preds, average="macro")
    print(f"\n  OOF macro-F1 = {overall:.4f}  (mean fold = {np.mean(fold_scores):.4f})")
    print("\n  Classification report (OOF):")
    print(classification_report(y, oof_preds, digits=4))
    return overall


# ---------------------------------------------------------------------------
# Train final model + predict
# ---------------------------------------------------------------------------
def train_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Train on full training set, predict on test set."""
    X_train = train[TEXT_COL].apply(preprocess)
    y_train = train[TARGET_COL].values
    X_test = test[TEXT_COL].apply(preprocess)

    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------
def make_submission(test: pd.DataFrame, preds: np.ndarray, submission_template: pd.DataFrame):
    out = submission_template.copy()
    pred_map = dict(zip(test["ID"], preds))
    out[TARGET_COL] = out["ID"].map(pred_map)

    assert out[TARGET_COL].isnull().sum() == 0, "Some test IDs missing from predictions!"
    assert out[TARGET_COL].isin(BIRADS_CLASSES).all(), "Predictions contain invalid BI-RADS values!"

    out_path = DATA_DIR / "submission_output.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}")
    print(out.head(10))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    train, test, submission = load_data()
    print(f"  Train: {len(train)} | Test: {len(test)} | Submission IDs: {len(submission)}")

    print(f"\nRunning {N_FOLDS}-fold CV...")
    cross_validate(train)

    print("\nTraining final model on full data...")
    preds = train_and_predict(train, test)

    print("\nGenerating submission...")
    make_submission(test, preds, submission)


if __name__ == "__main__":
    main()
