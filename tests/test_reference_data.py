"""Fast checks of the distributed feature tables; no NLP model downloads."""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_precomputed import ABLATIONS, FEATURE_GROUPS, load_features  # noqa: E402


class TestReferenceData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_features(ROOT / "outputs")

    def test_split_sizes(self):
        self.assertEqual({key: len(df) for key, df in self.tables.items()},
                         {"train": 6406, "dev": 796, "test": 811})

    def test_feature_groups(self):
        self.assertEqual([len(v) for v in FEATURE_GROUPS.values()], [9, 4, 7, 8])
        self.assertEqual(len(ABLATIONS), 7)
        self.assertEqual(len(set(ABLATIONS["all_groups"])), 28)

    def test_binary_labels_and_finite_features(self):
        for name, df in self.tables.items():
            with self.subTest(split=name):
                self.assertEqual(set(df["delta"].unique()), {0, 1})
                self.assertTrue(df.drop(columns="delta").apply(pd.to_numeric, errors="coerce").notna().all().all())

    def test_target_distribution(self):
        positives = sum(int(df["delta"].sum()) for df in self.tables.values())
        self.assertEqual(positives, 4192)


if __name__ == "__main__":
    unittest.main()
