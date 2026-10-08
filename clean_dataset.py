"""
clean_dataset.py

Clean and split the flat ChangeMyView persuasion dataset created from ConvoKit.

Expected input:
    data/changemyview_dataset.csv

Expected output:
    data/changemyview_dataset_clean.csv
    data/train.csv
    data/dev.csv
    data/test.csv

Main goals:
1. Remove leakage from post_text:
   - strip OP edit sections (e.g. EDIT:, Edit 2:, etc.)
   - remove moderator footer
2. Clean both post_text and comment_text:
   - decode HTML entities
   - simplify Markdown links
   - replace bare URLs with URL token
   - remove quote markers and superficial markdown artifacts
   - normalize whitespace
3. Filter poor-quality rows:
   - empty texts
   - deleted/removed placeholders
   - too-short posts/comments
4. Deduplicate rows
5. Split by post_id so that the same OP/thread never appears in multiple splits
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Set, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Config:
    """
    Configuration for dataset cleaning and splitting.
    """
    project_root: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent / "data"

    input_filename: str = "changemyview_dataset.csv"
    clean_filename: str = "changemyview_dataset_clean.csv"
    train_filename: str = "train.csv"
    dev_filename: str = "dev.csv"
    test_filename: str = "test.csv"

    post_id_col: str = "post_id"
    comment_id_col: str = "comment_id"
    post_text_col: str = "post_text"
    comment_text_col: str = "comment_text"
    target_col: str = "delta"

    # Split proportions
    train_size: float = 0.80
    dev_size: float = 0.10
    test_size: float = 0.10

    random_state: int = 42

    # Text filtering thresholds
    min_post_chars: int = 80
    min_comment_chars: int = 30
    min_post_words: int = 15
    min_comment_words: int = 5

    # Leakage / cleanup settings
    remove_post_edits: bool = True
    remove_comment_edits: bool = True
    remove_moderator_footer: bool = True
    replace_urls_with_token: bool = True

    @property
    def input_path(self) -> Path:
        return self.data_dir / self.input_filename

    @property
    def clean_path(self) -> Path:
        return self.data_dir / self.clean_filename

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_filename

    @property
    def dev_path(self) -> Path:
        return self.data_dir / self.dev_filename

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_filename


config = Config()


# =============================================================================
# 2. Constants and regexes
# =============================================================================

INVALID_TEXT_VALUES: Set[str] = {
    "",
    "nan",
    "none",
    "null",
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
}

# Examples matched:
# EDIT:
# Edit:
# EDIT 2:
# edit 3:
EDIT_PATTERN = re.compile(r"(?is)\bedit\s*\d*\s*:")

MOD_FOOTER_PATTERNS = [
    re.compile(r"(?is)_{3,}.*$"),
    re.compile(r"(?is)>\s*\*hello,\s*users\s*of\s*cmv!.*$"),
    re.compile(r"(?is)hello,\s*users\s*of\s*cmv!.*$"),
]

# Markdown links: [anchor](url) -> anchor
#MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\s*\((https?://[^)\s]+)\)"
)

# Bare URLs
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Markdown decoration
MARKDOWN_DECORATION_PATTERN = re.compile(r"[*_`]+")

# Strikethrough ~~text~~ -> text
STRIKETHROUGH_PATTERN = re.compile(r"~~(.*?)~~")

# Leading quote markers at line starts: > quoted text -> quoted text
LINE_QUOTE_MARKER_PATTERN = re.compile(r"(?m)^\s*>\s?")

# General whitespace normalization
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
MULTISPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?])")

# Sometimes repeated quotes remain ugly
DOUBLE_DOUBLE_QUOTES_PATTERN = re.compile(r'""+')

PAREN_URL_TOKEN_PATTERN = re.compile(r"\(\s*URL\s*\)")
LEFT_PAREN_URL_PATTERN = re.compile(r"\(\s*URL\b")
RIGHT_PAREN_AFTER_URL_PATTERN = re.compile(r"\bURL\s*\)")
INLINE_QUOTE_MARKER_PATTERN = re.compile(r"\s>\s*|\n>\s*")


# =============================================================================
# 3. Utility helpers
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    return text.strip()


def normalize_text(text: Any) -> str:
    """
    Basic normalization for arbitrary input:
    - convert to string
    - HTML-unescape
    - normalize whitespace
    """
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = text.strip()
    text = normalize_whitespace(text)
    return text


def is_invalid_placeholder_text(text: str) -> bool:
    """
    Check for clearly invalid placeholder strings.
    """
    return normalize_text(text).lower() in INVALID_TEXT_VALUES


def word_count(text: str) -> int:
    """
    Approximate word count.
    """
    return len(re.findall(r"\b\w+\b", text))


def remove_edit_sections(text: str) -> str:
    """
    Remove everything from the first EDIT-like marker onward.
    """
    match = EDIT_PATTERN.search(text)
    if match:
        return text[:match.start()].strip()
    return text


def remove_moderator_footer(text: str) -> str:
    """
    Remove standard CMV moderator footer if present.
    """
    cleaned = text
    for pattern in MOD_FOOTER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            cleaned = cleaned[:match.start()].strip()
    return cleaned


def simplify_markdown_links(text: str) -> str:
    """
    Convert Markdown links to anchor text only.

    Example:
        [BBC](http://example.com) -> BBC
    """
    return MARKDOWN_LINK_PATTERN.sub(r"\1", text)


def replace_bare_urls(text: str) -> str:
    """
    Replace bare URLs with a generic URL token.
    """
    if config.replace_urls_with_token:
        return URL_PATTERN.sub("URL", text)
    return URL_PATTERN.sub("", text)


def clean_url_artifacts(text: str) -> str:
    """
    Clean leftover punctuation artifacts around URL replacements.
    """
    text = PAREN_URL_TOKEN_PATTERN.sub(" URL ", text)
    text = LEFT_PAREN_URL_PATTERN.sub(" URL", text)
    text = RIGHT_PAREN_AFTER_URL_PATTERN.sub("URL", text)
    return text


def remove_strikethrough_markup(text: str) -> str:
    """
    Remove ~~ ~~ markers but keep enclosed text.
    """
    return STRIKETHROUGH_PATTERN.sub(r"\1", text)


def remove_line_quote_markers(text: str) -> str:
    """
    Remove leading '>' quote markers from the start of lines.
    """
    return LINE_QUOTE_MARKER_PATTERN.sub("", text)


def remove_inline_quote_markers(text: str) -> str:
    """
    Remove leftover inline quote markers such as ' > quoted text'
    after earlier normalization steps.
    """
    return INLINE_QUOTE_MARKER_PATTERN.sub(" ", text)


def strip_markdown_decorations(text: str) -> str:
    """
    Remove superficial markdown decoration characters.
    """
    return MARKDOWN_DECORATION_PATTERN.sub("", text)


def normalize_quotes(text: str) -> str:
    """
    Light normalization of doubled quote artifacts.
    """
    text = DOUBLE_DOUBLE_QUOTES_PATTERN.sub('"', text)
    return text


def clean_common_text(text: Any) -> str:
    """
    Common cleaning steps used for both post_text and comment_text.
    """
    text = normalize_text(text)
    text = simplify_markdown_links(text)
    text = replace_bare_urls(text)
    text = clean_url_artifacts(text)
    text = remove_strikethrough_markup(text)
    text = remove_line_quote_markers(text)
    text = remove_inline_quote_markers(text)
    text = strip_markdown_decorations(text)
    text = normalize_quotes(text)
    text = normalize_whitespace(text)
    return text


def clean_post_text(text: Any) -> str:
    """
    Clean original post text.

    Includes stronger leakage prevention:
    - remove edit sections
    - remove moderator footer
    """
    text = clean_common_text(text)

    if config.remove_post_edits:
        text = remove_edit_sections(text)

    if config.remove_moderator_footer:
        text = remove_moderator_footer(text)

    text = normalize_whitespace(text)
    return text


def clean_comment_text(text: Any) -> str:
    """
    Clean comment text.

    Includes optional removal of comment edit sections.
    """
    text = clean_common_text(text)

    if config.remove_comment_edits:
        text = remove_edit_sections(text)

    text = normalize_whitespace(text)
    return text


def row_passes_quality_filters(post_text: str, comment_text: str) -> bool:
    """
    Decide whether a row is good enough to keep.
    """
    if not post_text or not comment_text:
        return False

    if is_invalid_placeholder_text(post_text) or is_invalid_placeholder_text(comment_text):
        return False

    if len(post_text) < config.min_post_chars:
        return False

    if len(comment_text) < config.min_comment_chars:
        return False

    if word_count(post_text) < config.min_post_words:
        return False

    if word_count(comment_text) < config.min_comment_words:
        return False

    return True


# =============================================================================
# 4. Cleaning pipeline
# =============================================================================

def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist in the input dataframe.
    """
    required = {
        config.post_id_col,
        config.comment_id_col,
        config.post_text_col,
        config.comment_text_col,
        config.target_col,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw labeled dataset.
    """
    validate_input_columns(df)

    print("=" * 80)
    print("START CLEANING")
    print("=" * 80)
    print(f"Initial rows: {len(df)}")

    before = len(df)
    df = df[df[config.target_col].isin([0, 1])].copy()
    print(f"Removed rows with non-binary labels: {before - len(df)}")

    before = len(df)
    df = df.dropna(
        subset=[
            config.post_id_col,
            config.comment_id_col,
            config.post_text_col,
            config.comment_text_col,
            config.target_col,
        ]
    ).copy()
    print(f"Removed rows with missing required values: {before - len(df)}")

    # Preserve originals optionally for debugging if you ever want to inspect diffs.
    df[config.post_text_col] = df[config.post_text_col].apply(clean_post_text)
    df[config.comment_text_col] = df[config.comment_text_col].apply(clean_comment_text)

    before = len(df)
    quality_mask = df.apply(
        lambda row: row_passes_quality_filters(
            row[config.post_text_col],
            row[config.comment_text_col]
        ),
        axis=1
    )
    df = df[quality_mask].copy()
    print(f"Removed rows failing text-quality filters: {before - len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=[config.comment_id_col]).copy()
    print(f"Removed duplicate comment_id rows: {before - len(df)}")

    before = len(df)
    df = df.drop_duplicates(
        subset=[
            config.post_id_col,
            config.post_text_col,
            config.comment_text_col,
            config.target_col
        ]
    ).copy()
    print(f"Removed exact duplicate text triples: {before - len(df)}")

    df[config.target_col] = df[config.target_col].astype(int)

    print(f"Final cleaned rows: {len(df)}")
    print("=" * 80)

    return df


# =============================================================================
# 5. Group split by post_id
# =============================================================================

def split_by_post_id(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/dev/test by post_id.
    """
    total = config.train_size + config.dev_size + config.test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split proportions must sum to 1.0, got {total:.4f}")

    groups = df[config.post_id_col]

    temp_size = config.dev_size + config.test_size
    gss_1 = GroupShuffleSplit(
        n_splits=1,
        train_size=config.train_size,
        test_size=temp_size,
        random_state=config.random_state
    )

    train_idx, temp_idx = next(gss_1.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    dev_relative = config.dev_size / (config.dev_size + config.test_size)

    gss_2 = GroupShuffleSplit(
        n_splits=1,
        train_size=dev_relative,
        test_size=1.0 - dev_relative,
        random_state=config.random_state
    )

    temp_groups = temp_df[config.post_id_col]
    dev_idx_rel, test_idx_rel = next(gss_2.split(temp_df, groups=temp_groups))

    dev_df = temp_df.iloc[dev_idx_rel].copy()
    test_df = temp_df.iloc[test_idx_rel].copy()

    return train_df, dev_df, test_df


# =============================================================================
# 6. Diagnostics
# =============================================================================

def print_dataset_diagnostics(name: str, df: pd.DataFrame) -> None:
    """
    Print summary diagnostics for a dataset split.
    """
    print(f"\n{name.upper()} DIAGNOSTICS")
    print("-" * 80)
    print(f"Rows: {len(df)}")
    print(f"Unique post_id: {df[config.post_id_col].nunique()}")
    print(f"Unique comment_id: {df[config.comment_id_col].nunique()}")
    print(f"Label distribution: {df[config.target_col].value_counts(dropna=False).to_dict()}")
    print(f"Positive rate: {df[config.target_col].mean():.4f}")
    print("-" * 80)


def assert_no_group_overlap(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Assert that post_id groups do not overlap across splits.
    """
    train_posts = set(train_df[config.post_id_col])
    dev_posts = set(dev_df[config.post_id_col])
    test_posts = set(test_df[config.post_id_col])

    overlap_train_dev = train_posts & dev_posts
    overlap_train_test = train_posts & test_posts
    overlap_dev_test = dev_posts & test_posts

    if overlap_train_dev or overlap_train_test or overlap_dev_test:
        raise AssertionError("Found overlapping post_id values across splits.")

    print("\nVerified: no post_id overlap across train/dev/test splits.")


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    """
    Main execution flow.
    """
    print("Starting dataset cleaning pipeline...")
    config.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input dataset from: {config.input_path}")
    df = pd.read_csv(config.input_path)

    clean_df = clean_dataframe(df)

    clean_df.to_csv(config.clean_path, index=False)
    print(f"Saved cleaned dataset to: {config.clean_path}")

    train_df, dev_df, test_df = split_by_post_id(clean_df)
    assert_no_group_overlap(train_df, dev_df, test_df)

    train_df.to_csv(config.train_path, index=False)
    dev_df.to_csv(config.dev_path, index=False)
    test_df.to_csv(config.test_path, index=False)

    print(f"Saved train split to: {config.train_path}")
    print(f"Saved dev split to:   {config.dev_path}")
    print(f"Saved test split to:  {config.test_path}")

    print_dataset_diagnostics("clean_full", clean_df)
    print_dataset_diagnostics("train", train_df)
    print_dataset_diagnostics("dev", dev_df)
    print_dataset_diagnostics("test", test_df)

    print("\nDone.")


if __name__ == "__main__":
    main()