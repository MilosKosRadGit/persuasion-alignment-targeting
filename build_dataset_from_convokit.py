"""
build_dataset_from_convokit.py

Build a flat CSV dataset from the ConvoKit Winning Arguments corpus for
feature-based persuasion modeling.

Target output schema:
    post_id, comment_id, post_text, comment_text, delta

Expected project structure:
    project/
    ├── data/
    ├── outputs/
    ├── build_dataset_from_convokit.py
    └── persuasion_v1_pipeline.py

This script:
1. downloads the ConvoKit "winning-args-corpus",
2. extracts OP text from conversation metadata,
3. extracts top-level replies as candidate persuasive comments,
4. assigns the binary persuasion label from utterance meta["success"],
5. saves a flat CSV into data/changemyview_dataset.csv

Date: 2026-03-31
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from convokit import Corpus, download


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Config:
    """
    Configuration for dataset construction.
    """
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent / "data"
    output_filename: str = "changemyview_dataset.csv"

    # Correct ConvoKit corpus name from official documentation
    corpus_name: str = "winning-args-corpus"

    # Recommended for V1: keep only direct replies to OP
    only_top_level_replies: bool = True

    # Keep only rows whose success label is explicitly 0 or 1
    keep_only_labeled_rows: bool = True

    # Drop rows with empty extracted text
    drop_empty_texts: bool = True

    # Number of sample rows to preview
    preview_rows: int = 5

    @property
    def output_path(self) -> Path:
        return self.data_dir / self.output_filename


config = Config()


# =============================================================================
# 2. Utility helpers
# =============================================================================

def normalize_text(text: Any) -> str:
    """
    Normalize text into a stripped single-space string.
    """
    if text is None:
        return ""
    text = str(text).strip()
    text = " ".join(text.split())
    return text


def build_op_text_from_conversation(convo) -> str:
    """
    Construct OP text from conversation metadata.

    Official docs state that conversation metadata includes:
    - op-title
    - op-text-body

    We combine them into one text field for downstream modeling.
    """
    title = normalize_text(convo.meta.get("op-title"))
    body = normalize_text(convo.meta.get("op-text-body"))

    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def extract_success_label(utt) -> Optional[int]:
    """
    Extract the persuasion outcome from utterance metadata.

    Official docs for the Winning Arguments corpus state that utterance metadata
    includes 'success':
    - 1 if part of a successful argument thread
    - 0 if part of an unsuccessful paired thread
    - None otherwise
    """
    value = utt.meta.get("success", None)

    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    value_str = str(value).strip().lower()
    if value_str in {"1", "true"}:
        return 1
    if value_str in {"0", "false"}:
        return 0

    return None


def is_top_level_reply(utt, utt_by_id: Dict[str, Any]) -> bool:
    """
    Return True if the utterance directly replies to the original post.
    """
    if utt.reply_to is None:
        return False

    parent = utt_by_id.get(utt.reply_to)
    if parent is None:
        return False

    # In this corpus, the root post has reply_to == None
    return parent.reply_to is None


def build_id_to_utterance(corpus: Corpus) -> Dict[str, Any]:
    """
    Build a lookup from utterance ID to utterance object.
    """
    return {utt.id: utt for utt in corpus.iter_utterances()}


# =============================================================================
# 3. Core extraction
# =============================================================================

def extract_examples_from_corpus(corpus: Corpus) -> pd.DataFrame:
    """
    Extract a flat dataframe with columns:
        post_id, comment_id, post_text, comment_text, delta

    Strategy:
    - iterate over utterances,
    - keep replies (not root posts),
    - optionally keep only top-level replies,
    - get OP text from conversation metadata,
    - get label from utt.meta["success"].
    """
    utt_by_id = build_id_to_utterance(corpus)
    rows: List[Dict[str, Any]] = []

    total_utts = 0
    kept = 0
    skipped_unlabeled = 0

    for utt in corpus.iter_utterances():
        total_utts += 1

        # Skip root posts
        if utt.reply_to is None:
            continue

        if config.only_top_level_replies and not is_top_level_reply(utt, utt_by_id):
            continue

        success = extract_success_label(utt)

        if config.keep_only_labeled_rows and success is None:
            skipped_unlabeled += 1
            continue

        convo = corpus.get_conversation(utt.conversation_id)
        post_text = build_op_text_from_conversation(convo)
        comment_text = normalize_text(utt.text)

        if config.drop_empty_texts and (not post_text or not comment_text):
            continue

        rows.append(
            {
                "post_id": str(utt.conversation_id),
                "comment_id": str(utt.id),
                "post_text": post_text,
                "comment_text": comment_text,
                "delta": int(success) if success is not None else None,
            }
        )
        kept += 1

        if kept % 5000 == 0:
            print(f"Kept {kept} labeled examples so far...")

    print(f"Total utterances scanned: {total_utts}")
    print(f"Labeled examples kept: {kept}")
    print(f"Unlabeled eligible replies skipped: {skipped_unlabeled}")

    return pd.DataFrame(rows)


# =============================================================================
# 4. Diagnostics
# =============================================================================

def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by comment_id.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["comment_id"]).copy()
    after = len(df)
    print(f"Deduplicated rows removed: {before - after}")
    return df


def print_diagnostics(df: pd.DataFrame) -> None:
    """
    Print useful diagnostics about the extracted dataset.
    """
    print("=" * 80)
    print("DATASET DIAGNOSTICS")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Unique post_id: {df['post_id'].nunique()}")
    print(f"Unique comment_id: {df['comment_id'].nunique()}")

    if len(df) > 0:
        counts = df["delta"].value_counts(dropna=False).to_dict()
        print(f"Delta distribution: {counts}")
        if df["delta"].notna().any():
            print(f"Positive rate: {df['delta'].mean():.4f}")

    print("\nPreview:")
    print(df.head(config.preview_rows).to_string(index=False))
    print("=" * 80)


# =============================================================================
# 5. Main
# =============================================================================

def main() -> None:
    """
    Main entry point.
    """
    print("Starting dataset build from ConvoKit...")
    config.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/loading corpus: {config.corpus_name}")
    corpus = Corpus(filename=download(config.corpus_name))

    print("Corpus loaded successfully.")
    print("Extracting flat dataset...")

    df = extract_examples_from_corpus(corpus)
    df = deduplicate_rows(df)

    # Keep only explicit binary labels for the main V1 task
    df = df[df["delta"].isin([0, 1])].copy()

    print_diagnostics(df)

    df.to_csv(config.output_path, index=False)
    print(f"Saved dataset to: {config.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()