"""Synthetic, locally reproducible tests for the cleaning/splitting functions."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clean_dataset import (  # noqa: E402
    assert_no_group_overlap,
    clean_comment_text,
    clean_post_text,
    split_by_post_id,
)


class TestCleaning(unittest.TestCase):
    def test_posthoc_edit_removed(self):
        post = "Original view with several explanations. EDIT: I changed my opinion."
        cleaned = clean_post_text(post)
        self.assertIn("Original view", cleaned)
        self.assertNotIn("changed my opinion", cleaned)

    def test_group_split_no_post_overlap(self):
        rows = [
            {"post_id": f"post_{post}", "comment_id": f"comment_{post}_{i}", "delta": i % 2}
            for post in range(35)
            for i in range(3)
        ]
        df = pd.DataFrame(rows)
        train, dev, test = split_by_post_id(df)
        assert_no_group_overlap(train, dev, test)
        self.assertEqual(len(train) + len(dev) + len(test), len(df))

    def test_markdown_link_simplified(self):
        result = clean_comment_text("See [source](https://example.com/path) for more information.")
        self.assertIn("source", result)
        self.assertNotIn("https://example.com/path", result)


if __name__ == "__main__":
    unittest.main()
