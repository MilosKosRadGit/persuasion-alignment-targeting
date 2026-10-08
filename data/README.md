# Local dataset files

The following CSV files are deliberately **not distributed in the public repository** because they contain user-generated Reddit text obtained via the ConvoKit *Winning Arguments* corpus:

- `changemyview_dataset.csv` (original derived flat dataset)
- `changemyview_dataset_clean.csv` (cleaned dataset)
- `train.csv`, `dev.csv`, `test.csv` (group-separated subsets)

To generate them, run the first two scripts described in the root README. The `data/` folder is git-ignored apart from this documentation. Keep local copies of the files for full reproduction, but check source licensing/terms before redistributing full texts.

The `outputs/*_features_v1.csv` tables intentionally contain only aggregate numeric features and the binary `delta` label, rather than full Reddit posts. They are included as a compact reproducibility artifact; their schema is checked when running the evaluation script.
