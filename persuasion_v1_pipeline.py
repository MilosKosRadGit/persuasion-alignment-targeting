"""
persuasion_v1_pipeline.py

Version 1 research pipeline for feature-based prediction of successful persuasion
in online argumentation datasets such as ChangeMyView.

Main unit of analysis:
    (OP_text, comment_text, delta)

This script:
1. loads precomputed dataset splits from a sibling `data/` folder,
2. extracts interpretable feature groups separately for train/dev/test,
3. saves feature tables to `outputs/`,
4. trains baseline models on train,
5. evaluates on dev and test,
6. optionally runs simple ablation experiments by feature group.

Date: 2026-03-31
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Config:
    """
    Global configuration for the persuasion pipeline.
    """
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent / "data"
    output_dir: Path = Path(__file__).resolve().parent / "outputs"

    train_filename: str = "train.csv"
    dev_filename: str = "dev.csv"
    test_filename: str = "test.csv"

    text_col_op: str = "post_text"
    text_col_comment: str = "comment_text"
    target_col: str = "delta"

    sentence_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    spacy_model_name: str = "en_core_web_sm"

    targeting_threshold: float = 0.50
    random_state: int = 42
    progress_every: int = 100
    run_ablation: bool = True

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_filename

    @property
    def dev_path(self) -> Path:
        return self.data_dir / self.dev_filename

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_filename

    @property
    def train_feature_output_path(self) -> Path:
        return self.output_dir / "train_features_v1.csv"

    @property
    def dev_feature_output_path(self) -> Path:
        return self.output_dir / "dev_features_v1.csv"

    @property
    def test_feature_output_path(self) -> Path:
        return self.output_dir / "test_features_v1.csv"


config = Config()


# =============================================================================
# 2. Marker lexicons
# =============================================================================

EVIDENCE_MARKERS = {
    "because", "since", "for example", "for instance", "evidence", "data",
    "research", "study", "studies", "statistics", "according to"
}

REASONING_MARKERS = {
    "therefore", "thus", "hence", "so", "if", "then", "which means",
    "this implies", "as a result", "consequently"
}

CONCESSION_MARKERS = {
    "you are right", "you're right", "i agree", "fair point", "i understand",
    "i see why", "that makes sense", "while i agree", "although i agree"
}

CONTRAST_MARKERS = {
    "however", "but", "although", "though", "yet", "on the other hand",
    "at the same time", "while"
}

HEDGE_MARKERS = {
    "maybe", "perhaps", "might", "could", "seems", "appears",
    "i think", "i believe", "in my view", "possibly"
}

CERTAINTY_MARKERS = {
    "clearly", "obviously", "definitely", "certainly", "undoubtedly",
    "there is no doubt", "it is clear"
}

EMPATHY_MARKERS = {
    "i understand", "i get why", "i see why", "that makes sense",
    "i can see", "i sympathize", "it is understandable"
}

POLITENESS_MARKERS = {
    "please", "thank you", "thanks", "appreciate", "respectfully"
}

TOXICITY_LEXICON = {
    "idiot", "stupid", "dumb", "moron", "nonsense", "ridiculous", "insane"
}


# =============================================================================
# 3. NLP model initialization
# =============================================================================

nlp = spacy.load(config.spacy_model_name)
embed_model = SentenceTransformer(config.sentence_model_name)


# =============================================================================
# 4. Text utility functions
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize raw text by stripping leading/trailing spaces and collapsing
    repeated whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def get_sentences(text: str) -> List[str]:
    """
    Segment text into sentences using spaCy.
    """
    text = normalize_text(text)
    if not text:
        return []
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def get_tokens(text: str) -> List[str]:
    """
    Tokenize text with spaCy.
    """
    text = normalize_text(text)
    if not text:
        return []
    doc = nlp(text)
    return [tok.text.lower() for tok in doc if not tok.is_space]


def get_alpha_tokens(text: str) -> List[str]:
    """
    Extract lemmatized alphabetic content tokens.
    """
    text = normalize_text(text)
    if not text:
        return []
    doc = nlp(text)
    return [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]


def safe_divide(a: float, b: float) -> float:
    """
    Safely divide a by b.
    """
    return a / b if b != 0 else 0.0


def count_phrase_markers(text: str, markers: Sequence[str]) -> int:
    """
    Count approximate occurrences of marker phrases in text.
    """
    text_lower = normalize_text(text).lower()
    count = 0
    for marker in markers:
        count += text_lower.count(marker)
    return count


def jaccard_overlap(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Compute Jaccard overlap between two token lists.
    """
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# =============================================================================
# 5. Feature extraction functions
# =============================================================================

def extract_length_features(comment_text: str) -> Dict[str, float]:
    """
    Extract basic length and structural features from the comment text.
    """
    comment_text = normalize_text(comment_text)
    sentences = get_sentences(comment_text)
    tokens = get_tokens(comment_text)
    alpha_tokens = get_alpha_tokens(comment_text)

    num_chars = len(comment_text)
    num_words = len([t for t in tokens if re.search(r"\w", t)])
    num_sentences = len(sentences)
    avg_sentence_length = safe_divide(num_words, num_sentences)

    return {
        "comment_num_chars": float(num_chars),
        "comment_num_words": float(num_words),
        "comment_num_sentences": float(num_sentences),
        "comment_avg_sentence_length": float(avg_sentence_length),
        "comment_num_content_words": float(len(alpha_tokens)),
    }


def extract_overlap_features(op_text: str, comment_text: str) -> Dict[str, float]:
    """
    Extract lexical overlap features between OP text and comment text.
    """
    op_tokens = get_alpha_tokens(op_text)
    comment_tokens = get_alpha_tokens(comment_text)

    overlap = jaccard_overlap(op_tokens, comment_tokens)
    shared = len(set(op_tokens) & set(comment_tokens))

    return {
        "lexical_jaccard_overlap": float(overlap),
        "lexical_shared_content_words": float(shared),
        "lexical_shared_ratio_vs_comment": float(
            safe_divide(shared, len(set(comment_tokens)))
        ),
    }


def extract_embedding_similarity(op_text: str, comment_text: str) -> Dict[str, float]:
    """
    Compute coarse semantic similarity between OP and comment.
    """
    op_text = normalize_text(op_text)
    comment_text = normalize_text(comment_text)

    if not op_text or not comment_text:
        return {"op_comment_embedding_cosine": 0.0}

    op_emb = embed_model.encode(op_text, convert_to_tensor=True)
    comment_emb = embed_model.encode(comment_text, convert_to_tensor=True)

    cosine = util.cos_sim(op_emb, comment_emb).item()

    return {
        "op_comment_embedding_cosine": float(cosine)
    }


def extract_targeting_features(
    op_text: str,
    comment_text: str,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Estimate how well the comment targets the OP.
    """
    op_sentences = get_sentences(op_text)
    comment_sentences = get_sentences(comment_text)

    if len(op_sentences) == 0 or len(comment_sentences) == 0:
        return {
            "targeting_mean_max": 0.0,
            "targeting_max": 0.0,
            "targeting_coverage_05": 0.0,
            "targeting_top3_mean": 0.0,
        }

    op_emb = embed_model.encode(op_sentences, convert_to_tensor=True)
    comment_emb = embed_model.encode(comment_sentences, convert_to_tensor=True)

    sim_matrix = util.cos_sim(op_emb, comment_emb).cpu().numpy()
    max_per_op_sentence = sim_matrix.max(axis=1)

    targeting_mean_max = float(np.mean(max_per_op_sentence))
    targeting_max = float(np.max(max_per_op_sentence))
    targeting_coverage = float(np.mean(max_per_op_sentence >= threshold))

    top_k = min(3, len(max_per_op_sentence))
    top3_mean = float(np.mean(np.sort(max_per_op_sentence)[-top_k:]))

    return {
        "targeting_mean_max": targeting_mean_max,
        "targeting_max": targeting_max,
        "targeting_coverage_05": targeting_coverage,
        "targeting_top3_mean": top3_mean,
    }


def extract_argument_discourse_features(comment_text: str) -> Dict[str, float]:
    """
    Extract argumentation and discourse marker features from the comment.
    """
    evidence_count = count_phrase_markers(comment_text, EVIDENCE_MARKERS)
    reasoning_count = count_phrase_markers(comment_text, REASONING_MARKERS)
    concession_count = count_phrase_markers(comment_text, CONCESSION_MARKERS)
    contrast_count = count_phrase_markers(comment_text, CONTRAST_MARKERS)
    hedge_count = count_phrase_markers(comment_text, HEDGE_MARKERS)
    certainty_count = count_phrase_markers(comment_text, CERTAINTY_MARKERS)

    return {
        "evidence_marker_count": float(evidence_count),
        "reasoning_marker_count": float(reasoning_count),
        "concession_marker_count": float(concession_count),
        "contrast_marker_count": float(contrast_count),
        "hedge_marker_count": float(hedge_count),
        "certainty_marker_count": float(certainty_count),
        "hedge_certainty_ratio": float(safe_divide(hedge_count, certainty_count + 1)),
    }


def extract_pragmatic_features(comment_text: str) -> Dict[str, float]:
    """
    Extract social/pragmatic features.
    """
    tokens = [t.lower() for t in get_tokens(comment_text)]
    total_tokens = len(tokens)

    empathy_count = count_phrase_markers(comment_text, EMPATHY_MARKERS)
    politeness_count = count_phrase_markers(comment_text, POLITENESS_MARKERS)
    toxicity_count = count_phrase_markers(comment_text, TOXICITY_LEXICON)

    first_person = sum(t in {"i", "me", "my", "mine"} for t in tokens)
    second_person = sum(t in {"you", "your", "yours"} for t in tokens)

    return {
        "empathy_marker_count": float(empathy_count),
        "politeness_marker_count": float(politeness_count),
        "toxicity_lexicon_count": float(toxicity_count),
        "first_person_ratio": float(safe_divide(first_person, total_tokens)),
        "second_person_ratio": float(safe_divide(second_person, total_tokens)),
    }


def extract_evidential_richness_features(comment_text: str) -> Dict[str, float]:
    """
    Extract surface indicators of evidential richness such as URLs, numbers,
    and year mentions.

    Important:
    cleaned text may contain a normalized URL token ("URL") instead of raw links,
    so we count both raw URL strings and the token itself.
    """
    text = normalize_text(comment_text)

    raw_url_count = len(re.findall(r"https?://\S+|www\.\S+", text))
    normalized_url_token_count = len(re.findall(r"\bURL\b", text, flags=re.IGNORECASE))
    num_urls = raw_url_count + normalized_url_token_count

    num_numbers = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    num_years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))

    return {
        "num_urls": float(num_urls),
        "num_numbers": float(num_numbers),
        "num_years": float(num_years),
    }


def extract_features_for_pair(op_text: str, comment_text: str) -> Dict[str, float]:
    """
    Master feature extraction function for one (OP, comment) pair.
    """
    op_text = normalize_text(op_text)
    comment_text = normalize_text(comment_text)

    features: Dict[str, float] = {}
    features.update(extract_length_features(comment_text))
    features.update(extract_overlap_features(op_text, comment_text))
    features.update(extract_embedding_similarity(op_text, comment_text))
    features.update(
        extract_targeting_features(
            op_text,
            comment_text,
            threshold=config.targeting_threshold
        )
    )
    features.update(extract_argument_discourse_features(comment_text))
    features.update(extract_pragmatic_features(comment_text))
    features.update(extract_evidential_richness_features(comment_text))

    return features


# =============================================================================
# 6. Feature groups for ablation
# =============================================================================

FEATURE_GROUPS = {
    "surface": [
        "comment_num_chars",
        "comment_num_words",
        "comment_num_sentences",
        "comment_avg_sentence_length",
        "comment_num_content_words",
        "lexical_jaccard_overlap",
        "lexical_shared_content_words",
        "lexical_shared_ratio_vs_comment",
        "op_comment_embedding_cosine",
    ],
    "targeting": [
        "targeting_mean_max",
        "targeting_max",
        "targeting_coverage_05",
        "targeting_top3_mean",
    ],
    "discourse": [
        "evidence_marker_count",
        "reasoning_marker_count",
        "concession_marker_count",
        "contrast_marker_count",
        "hedge_marker_count",
        "certainty_marker_count",
        "hedge_certainty_ratio",
    ],
    "pragmatic": [
        "empathy_marker_count",
        "politeness_marker_count",
        "toxicity_lexicon_count",
        "first_person_ratio",
        "second_person_ratio",
        "num_urls",
        "num_numbers",
        "num_years",
    ],
}


# =============================================================================
# 7. Dataset-to-feature-table conversion
# =============================================================================

def validate_required_columns(df: pd.DataFrame, split_name: str) -> None:
    """
    Validate required columns in an input dataframe.
    """
    required_cols = {
        config.text_col_op,
        config.text_col_comment,
        config.target_col
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {split_name}: {missing}")


def build_feature_dataframe(
    df: pd.DataFrame,
    op_col: str,
    comment_col: str,
    target_col: str,
    split_name: str
) -> pd.DataFrame:
    """
    Convert one split into a numeric feature dataframe.
    """
    rows = []

    for idx, row in df.iterrows():
        op_text = row[op_col]
        comment_text = row[comment_col]
        delta = row[target_col]

        feat = extract_features_for_pair(op_text, comment_text)
        feat[target_col] = delta
        rows.append(feat)

        if (idx + 1) % config.progress_every == 0:
            print(f"[{split_name}] Processed {idx + 1} rows")

    return pd.DataFrame(rows)


# =============================================================================
# 8. Modeling and evaluation
# =============================================================================

def build_model(model_name: str):
    """
    Construct a model pipeline by name.
    """
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=config.random_state
            ))
        ])
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=config.random_state,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def evaluate_on_split(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    split_name: str
) -> None:
    """
    Evaluate a fitted model on one split.
    """
    preds = model.predict(X_eval)

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_eval)[:, 1]

    print(f"\n--- Evaluation on {split_name.upper()} ---")
    print(classification_report(y_eval, preds, digits=4))

    if probs is not None:
        auc = roc_auc_score(y_eval, probs)
        print(f"ROC-AUC: {auc:.4f}")


def train_and_evaluate_models(
    train_feature_df: pd.DataFrame,
    dev_feature_df: pd.DataFrame,
    test_feature_df: pd.DataFrame,
    target_col: str = "delta"
) -> None:
    """
    Train models on train and evaluate on dev and test.
    """
    X_train = train_feature_df.drop(columns=[target_col])
    y_train = train_feature_df[target_col].astype(int)

    X_dev = dev_feature_df.drop(columns=[target_col])
    y_dev = dev_feature_df[target_col].astype(int)

    X_test = test_feature_df.drop(columns=[target_col])
    y_test = test_feature_df[target_col].astype(int)

    for model_name in ["logreg", "rf"]:
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL: {model_name.upper()}")
        print("=" * 80)

        model = build_model(model_name)
        model.fit(X_train, y_train)

        evaluate_on_split(model, X_dev, y_dev, split_name="dev")
        evaluate_on_split(model, X_test, y_test, split_name="test")

        if model_name == "rf":
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
            importances = importances.sort_values(ascending=False).head(20)
            print("\nTop 20 feature importances:")
            print(importances.to_string())


def run_ablation_study(
    train_feature_df: pd.DataFrame,
    dev_feature_df: pd.DataFrame,
    target_col: str = "delta"
) -> None:
    """
    Run a simple ablation study using feature groups.

    Important:
    In this version, ablation is run on TRAIN -> DEV.
    We do not repeatedly use TEST for ablation decisions.
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY (TRAIN -> DEV)")
    print("=" * 80)

    y_train = train_feature_df[target_col].astype(int)
    y_dev = dev_feature_df[target_col].astype(int)

    ablation_sets = {
        "surface_only": FEATURE_GROUPS["surface"],
        "targeting_only": FEATURE_GROUPS["targeting"],
        "discourse_only": FEATURE_GROUPS["discourse"],
        "pragmatic_only": FEATURE_GROUPS["pragmatic"],
        "surface_plus_targeting": FEATURE_GROUPS["surface"] + FEATURE_GROUPS["targeting"],
        "surface_targeting_discourse": (
            FEATURE_GROUPS["surface"] +
            FEATURE_GROUPS["targeting"] +
            FEATURE_GROUPS["discourse"]
        ),
        "all_groups": (
            FEATURE_GROUPS["surface"] +
            FEATURE_GROUPS["targeting"] +
            FEATURE_GROUPS["discourse"] +
            FEATURE_GROUPS["pragmatic"]
        ),
    }

    for setting_name, feature_list in ablation_sets.items():
        X_train = train_feature_df[feature_list]
        X_dev = dev_feature_df[feature_list]

        model = build_model("logreg")
        model.fit(X_train, y_train)

        preds = model.predict(X_dev)
        probs = model.predict_proba(X_dev)[:, 1]
        auc = roc_auc_score(y_dev, probs)

        print(f"\n--- Ablation setting: {setting_name} ---")
        print(classification_report(y_dev, preds, digits=4))
        print(f"ROC-AUC: {auc:.4f}")


# =============================================================================
# 9. Main execution
# =============================================================================

def main() -> None:
    """
    Main execution entry point.

    Steps:
        1. Create output directory if needed.
        2. Load train/dev/test datasets from sibling data/ folder.
        3. Validate required columns.
        4. Drop rows with missing required values in each split.
        5. Extract feature tables separately for train/dev/test.
        6. Save feature tables to outputs/.
        7. Train on train, evaluate on dev and test.
        8. Optionally run ablation on train -> dev.
    """
    print("Starting persuasion V1 pipeline...")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading TRAIN dataset from: {config.train_path}")
    train_df = pd.read_csv(config.train_path)

    print(f"Loading DEV dataset from: {config.dev_path}")
    dev_df = pd.read_csv(config.dev_path)

    print(f"Loading TEST dataset from: {config.test_path}")
    test_df = pd.read_csv(config.test_path)

    validate_required_columns(train_df, "train")
    validate_required_columns(dev_df, "dev")
    validate_required_columns(test_df, "test")

    train_df = train_df.dropna(
        subset=[config.text_col_op, config.text_col_comment, config.target_col]
    ).copy()
    dev_df = dev_df.dropna(
        subset=[config.text_col_op, config.text_col_comment, config.target_col]
    ).copy()
    test_df = test_df.dropna(
        subset=[config.text_col_op, config.text_col_comment, config.target_col]
    ).copy()

    print(f"Rows in TRAIN after dropping missing values: {len(train_df)}")
    print(f"Rows in DEV after dropping missing values: {len(dev_df)}")
    print(f"Rows in TEST after dropping missing values: {len(test_df)}")

    train_feature_df = build_feature_dataframe(
        df=train_df,
        op_col=config.text_col_op,
        comment_col=config.text_col_comment,
        target_col=config.target_col,
        split_name="train"
    )

    dev_feature_df = build_feature_dataframe(
        df=dev_df,
        op_col=config.text_col_op,
        comment_col=config.text_col_comment,
        target_col=config.target_col,
        split_name="dev"
    )

    test_feature_df = build_feature_dataframe(
        df=test_df,
        op_col=config.text_col_op,
        comment_col=config.text_col_comment,
        target_col=config.target_col,
        split_name="test"
    )

    train_feature_df.to_csv(config.train_feature_output_path, index=False)
    dev_feature_df.to_csv(config.dev_feature_output_path, index=False)
    test_feature_df.to_csv(config.test_feature_output_path, index=False)

    print(f"Saved TRAIN feature table to: {config.train_feature_output_path}")
    print(f"Saved DEV feature table to:   {config.dev_feature_output_path}")
    print(f"Saved TEST feature table to:  {config.test_feature_output_path}")

    train_and_evaluate_models(
        train_feature_df=train_feature_df,
        dev_feature_df=dev_feature_df,
        test_feature_df=test_feature_df,
        target_col=config.target_col
    )

    if config.run_ablation:
        run_ablation_study(
            train_feature_df=train_feature_df,
            dev_feature_df=dev_feature_df,
            target_col=config.target_col
        )

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()