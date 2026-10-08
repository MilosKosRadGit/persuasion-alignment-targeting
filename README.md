# What Makes Arguments Persuasive?

**An Interpretable Analysis of Alignment and Targeted Engagement in Online Discussions**

This repository contains the feature-based experimental pipeline for predicting successful persuasion in Reddit **r/ChangeMyView** discussions. It uses the ConvoKit **Winning Arguments** corpus and investigates how global semantic/lexical **alignment**, sentence-level **targeting**, discourse markers, and pragmatic cues relate to the binary persuasion label (`delta`).

The goal is not only predictive performance, but an inspectable analysis of associations between linguistic signals and persuasive outcomes. The reported results do **not** establish causality. Sentence similarity is a **proxy for targeted engagement**, not a direct measure of claim rebuttal or argument quality.

## Repository contents

```text
.
├── build_dataset_from_convokit.py    # Extracts top-level labeled CMV replies into one CSV
├── clean_dataset.py                  # Cleans text, deduplicates, splits by original post
├── persuasion_v1_pipeline.py         # Original feature extraction + models + ablations
├── evaluate_precomputed.py           # Lightweight reproduction of published model results
├── requirements.txt                  # Dependencies for the full workflow
├── requirements-evaluation.txt       # Dependencies for evaluation from existing features
├── data/
│   └── README.md                      # Dataset instructions (full texts excluded from Git)
├── outputs/
│   ├── train_features_v1.csv          # Numeric features and binary label, 6,406 rows
│   ├── dev_features_v1.csv            # 796 rows
│   └── test_features_v1.csv           # 811 rows
├── results/
│   ├── reference_metrics.json         # Expected results reproduced from feature CSVs
│   ├── reference_ablation.csv          # Train-to-dev ablations
│   └── reference_rf_importances.csv   # Random Forest impurity feature importance
├── tests/
│   └── test_reference_data.py
└── README.sr.md
```

The main three scripts are retained from the experimental version used for the research work. No model/feature definition, split rule, or evaluation configuration has been modified to manufacture new results. `evaluate_precomputed.py` is an additional convenience script that deliberately reproduces the same training/evaluation choices without loading sentence-embedding or spaCy models.

## Quick reproduction (no text corpus download)

Python 3.10–3.12 is a practical starting choice for the full pipeline; individual newer Python versions may require compatible PyTorch/ConvoKit wheels. The **feature-only path** needs only NumPy, pandas and scikit-learn.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-evaluation.txt
python evaluate_precomputed.py
python -m unittest discover -s tests -v
```

On macOS/Linux, replace the virtual-environment commands with `python3 -m venv .venv` and `source .venv/bin/activate`.

This loads the **included numeric feature CSVs**, refits Logistic Regression and Random Forest, evaluates train-fitted models on both development and test sets, and runs seven Logistic Regression ablations **on development data only**. The script writes `results/metrics.json`, `results/ablation.csv`, and `results/rf_importances.csv` locally. Those regenerated files are ignored by Git; the `reference_*` files preserve a versioned example of expected results.

## Reconstructing everything from ConvoKit

For full reproduction, obtain the dependencies and spaCy English language model:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python build_dataset_from_convokit.py
python clean_dataset.py
python persuasion_v1_pipeline.py
```

1. `build_dataset_from_convokit.py` downloads `winning-args-corpus` using ConvoKit and saves `data/changemyview_dataset.csv`.
2. `clean_dataset.py` normalizes and filters texts, removes post-hoc `EDIT` sections to reduce leakage risk, and creates `data/changemyview_dataset_clean.csv` plus `data/train.csv`, `data/dev.csv`, `data/test.csv` with no overlapping original `post_id` values across splits.
3. `persuasion_v1_pipeline.py` loads these splits, extracts the features from scratch using spaCy and `sentence-transformers/all-mpnet-base-v2`, saves the three feature CSVs in `outputs/`, refits the models and reports evaluation/ablation results to the terminal.

**Note:** full feature extraction recalculates embeddings for all examples and can require significant compute and model downloads. Running the final script overwrites the existing `outputs/*_features_v1.csv` files. Differences in dependency or pretrained-model versions can affect regenerated features/results; the exact original package lockfile was not preserved.

## Dataset and splitting

- Extracted flat dataset: **8,106** labeled top-level replies.
- Clean dataset: **8,013** labeled replies across **3,021** original posts.
- Train: **6,406** replies; dev: **796**; test: **811**.
- Grouping key: original discussion `post_id`, to avoid responses to the same opening post appearing in different splits.
- Target `delta`: 1 means persuasive success as annotated by the CMV corpus; 0 means a negative example according to that source.

The local cleaning/splitting step is deterministic with `random_state=42`. See `data/README.md` for why full Reddit texts are not published here.

## Features and models

There are **28 numeric features** organized into four groups:

| Group | # | Description |
|---|---:|---|
| Surface | 9 | Response length/structure, lexical Jaccard and content-word overlap, overall post-response embedding cosine |
| Targeting | 4 | Aggregations of maximum sentence-to-sentence cosine similarities and post-sentence coverage |
| Discourse | 7 | Counts of reasoning/evidence/concession/contrast/hedging/certainty markers and a ratio |
| Pragmatic | 8 | Empathy/politeness/toxicity lexicons, pronouns, URLs, numbers, years |

Models:
- Logistic Regression: standardized numeric inputs; `class_weight="balanced"`, `max_iter=3000`, `random_state=42`.
- Random Forest: 300 trees; `class_weight="balanced"`, `random_state=42`.
- The development set is used for feature-group ablations. Reported test results are not the result of tuning multiple ablation settings on the test set.

## Reference results

| Model | Development ROC-AUC | Test ROC-AUC | Test accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.6152 | **0.6343** | 0.5845 |
| Random Forest | 0.5974 | 0.6081 | 0.5721 |

Logistic Regression development-set ablations (ROC-AUC): surface 0.6070; targeting 0.5083; discourse 0.5872; pragmatic 0.5443; surface+targeting 0.6098; surface+targeting+discourse 0.6111; all groups 0.6152. The change from surface alone to surface+targeting is **modest** (+0.0028 ROC-AUC), and no statistical significance or causal interpretation is asserted.

The results were reproduced by fitting new models from the included feature tables, independently of any saved models.

## Methodological limits

- The sentence-level targeting measures rely on cosine similarity; semantic correspondence is not proof that a reply addresses, refutes, or persuades about a specific claim.
- The outcome comes from the CMV corpus. It is observational and may reflect factors outside a single reply, including later conversational context.
- Lexicon-based discourse/pragmatic features are approximate. `count_phrase_markers` uses substring matching and can match short strings inside other words.
- Long texts can be truncated by the embedding model's input-length limits.
- There is **no directly evaluated fine-tuned transformer benchmark** on this identical split in the current experiments. Such comparisons remain future work.
- No formal uncertainty estimates or statistical significance tests are provided for differences between ablations.

## Source and data rights

This project processes third-party, user-generated content from the ConvoKit Winning Arguments dataset (based on Reddit ChangeMyView): https://convokit.cornell.edu/documentation/winning.html . Cite the dataset and ConvoKit if you reuse them. The full original/cleaned Reddit texts are not committed here. **If uploading through the GitHub website rather than `git`, use the separate public ZIP only:** `.gitignore` does not prevent manual web uploads. No software license is asserted by this repository; please establish licensing terms before public reuse or redistribution.

## Further development

Natural extensions include classical baselines (e.g., SVM and gradient boosting), fine-tuned transformer classifiers, feature-level error analysis, robustness/ablation checks, and calibrated measures of whether a reply directly engages a claim.
