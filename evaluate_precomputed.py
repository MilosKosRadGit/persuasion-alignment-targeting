"""Reproduce the original V1 model results from the numeric feature CSV files.

This module intentionally does not import the original feature extractor, as that
loads pretrained NLP models at import time. It mirrors the model configuration,
feature groups and train/dev/test evaluation protocol from the final pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
LABEL = "delta"
FEATURE_GROUPS = {
    "surface": [
        "comment_num_chars", "comment_num_words", "comment_num_sentences",
        "comment_avg_sentence_length", "comment_num_content_words",
        "lexical_jaccard_overlap", "lexical_shared_content_words",
        "lexical_shared_ratio_vs_comment", "op_comment_embedding_cosine",
    ],
    "targeting": [
        "targeting_mean_max", "targeting_max", "targeting_coverage_05",
        "targeting_top3_mean",
    ],
    "discourse": [
        "evidence_marker_count", "reasoning_marker_count", "concession_marker_count",
        "contrast_marker_count", "hedge_marker_count", "certainty_marker_count",
        "hedge_certainty_ratio",
    ],
    "pragmatic": [
        "empathy_marker_count", "politeness_marker_count", "toxicity_lexicon_count",
        "first_person_ratio", "second_person_ratio", "num_urls", "num_numbers",
        "num_years",
    ],
}
ABLATIONS = {
    "surface_only": FEATURE_GROUPS["surface"],
    "targeting_only": FEATURE_GROUPS["targeting"],
    "discourse_only": FEATURE_GROUPS["discourse"],
    "pragmatic_only": FEATURE_GROUPS["pragmatic"],
    "surface_plus_targeting": FEATURE_GROUPS["surface"] + FEATURE_GROUPS["targeting"],
    "surface_targeting_discourse": FEATURE_GROUPS["surface"] + FEATURE_GROUPS["targeting"] + FEATURE_GROUPS["discourse"],
    "all_groups": sum(FEATURE_GROUPS.values(), []),
}


def make_model(name: str):
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    raise ValueError(f"Unknown model: {name}")


def load_features(feature_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    cols = ABLATIONS["all_groups"] + [LABEL]
    for split in ("train", "dev", "test"):
        path = feature_dir / f"{split}_features_v1.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. See the README for rebuilding the full pipeline.")
        df = pd.read_csv(path)
        if df.columns.tolist() != cols:
            raise ValueError(
                f"Unexpected columns/order in {path}. Expected the original 28 features + delta."
                f" Got: {df.columns.tolist()}"
            )
        if df.isna().any().any():
            raise ValueError(f"Missing feature/label values in {path}")
        if not set(df[LABEL].unique()).issubset({0, 1}):
            raise ValueError(f"Nonbinary labels in {path}")
        tables[split] = df
    return tables


def evaluate(model: Any, df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    y = df[LABEL].astype(int)
    x = df[features]
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def run(feature_dir: Path = ROOT / "outputs", result_dir: Path = ROOT / "results") -> dict[str, Any]:
    tables = load_features(feature_dir)
    feature_cols = ABLATIONS["all_groups"]
    x_train = tables["train"][feature_cols]
    y_train = tables["train"][LABEL].astype(int)

    metrics: dict[str, Any] = {
        "counts": {split: len(table) for split, table in tables.items()},
        "models": {},
    }
    importance_rows: list[dict[str, Any]] = []
    for name in ("logreg", "rf"):
        model = make_model(name)
        model.fit(x_train, y_train)
        metrics["models"][name] = {
            split: evaluate(model, tables[split], feature_cols)
            for split in ("dev", "test")
        }
        if name == "rf":
            importance_rows = [
                {"feature": col, "importance": float(value)}
                for col, value in sorted(
                    zip(feature_cols, model.feature_importances_),
                    key=lambda item: item[1], reverse=True,
                )
            ]

    ablation_rows = []
    for name, features in ABLATIONS.items():
        model = make_model("logreg")
        model.fit(tables["train"][features], y_train)
        stats = evaluate(model, tables["dev"], features)
        ablation_rows.append({"setting": name, "n_features": len(features), **stats})

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")
    pd.DataFrame(ablation_rows).to_csv(result_dir / "ablation.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(result_dir / "rf_importances.csv", index=False)

    print("Model results (refitted from saved numeric features):")
    for name, scores in metrics["models"].items():
        print(f"  {name:6s} dev ROC-AUC={scores['dev']['roc_auc']:.4f}, "
              f"test ROC-AUC={scores['test']['roc_auc']:.4f}, "
              f"test accuracy={scores['test']['accuracy']:.4f}")
    print("Logistic regression ablations (development ROC-AUC):")
    for row in ablation_rows:
        print(f"  {row['setting']:29s} {row['roc_auc']:.4f}")
    print(f"Saved metrics and tables in {result_dir}")
    return {"metrics": metrics, "ablation": ablation_rows, "rf_importances": importance_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "outputs", help="Folder with numeric split CSVs")
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="Results directory")
    args = parser.parse_args()
    run(args.features, args.results)


if __name__ == "__main__":
    main()
