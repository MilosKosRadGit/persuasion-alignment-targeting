# Persuazija — uputstvo na srpskom

Repozitorijum sadrži eksperimentalni kod za rad **What Makes Arguments Persuasive? An Interpretable Analysis of Alignment and Targeted Engagement in Online Discussions**.

**Glavni README je na engleskom**, jer se rad i podaci odnose na međunarodnu NLP zajednicu. Ovde su najvažnije komande i napomene.

## Brza provera rezultata

U Windows PowerShell terminalu:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-evaluation.txt
python evaluate_precomputed.py
python -m unittest discover -s tests -v
```

Za ovu proveru **nije potrebno ponovo preuzimati Reddit tekstove, spaCy model ni embedding model**. Koriste se već sačuvane numeričke karakteristike u `outputs/`. Rezultati se upisuju u `results/`.

## Kompletan postupak

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python build_dataset_from_convokit.py
python clean_dataset.py
python persuasion_v1_pipeline.py
```

Redosled je: ConvoKit korpus → čišćenje i podela po originalnom `post_id` → izračunavanje 28 karakteristika → treniranje i evaluacija logističke regresije i Random Forest modela → ablacione analize. Treći korak može biti znatno sporiji zbog generisanja embeddinga. Ujedno može prepisati postojeće feature tabele.

## Podaci i GitHub

CSV fajlovi sa izvornim i očišćenim Reddit tekstovima **nisu deo javne GitHub verzije**. Lokalnu kopiju sa podacima treba čuvati odvojeno. U javnoj verziji su sačuvane numeričke tabele sa karakteristikama, kako bi se rezultati mogli brzo reprodukovati.