# BloodAI Medical Research

Research repository for **Intelligent Multi-Label Triage for Hematological Comorbidities using BERT**. The project contains the notebook workflow used to build a hybrid Synthea + MIMIC corpus, pre-train a BERT-style masked language model on medical tokens, fine-tune a multi-label triage classifier, and generate the evaluation figures used in the manuscript.

This repository is intended as a reproducible research artifact, not a clinical device. Outputs are for research review only and require independent clinical validation before any operational use.

## Authors

- Jan Banasik ([@JanBanasik](https://github.com/JanBanasik))
- Antoni Pater ([@antoniopater](https://github.com/antoniopater))

## Repository layout

- `notebooks_ethos/` - canonical notebook pipeline and small helper modules.
- `notebooks_ethos/outputs/plots/` - lightweight paper figures kept in Git.
- `notebooks_ethos/outputs/tables/` - lightweight paper tables kept in Git.
- `data/questions.json` - adaptive interview question bank.
- `data/lab_norms.json` - reference ranges used by the interview/tokenization flow.
- `Banasik_Pater_Paper_2026.pdf` - final paper manuscript.

Generated corpora, MIMIC/Synthea extracts, MLflow state, trained models, and checkpoints are intentionally excluded from Git.

## Notebook workflow

Run notebooks from the `notebooks_ethos/` directory, in order:

1. `00_Methodology.ipynb`
2. `01_Preprocessing_Synthea.ipynb`
3. `02_Preprocessing_MIMIC.ipynb`
4. `03_Merge_Datasets.ipynb`
5. `04_Train_MLM_Model.ipynb`
6. `05_Fine_Tune_Model.ipynb`
7. `06_Explainability.ipynb`
8. `07_Advanced_Evaluation.ipynb`
9. `08_Statistical_Analysis.ipynb`
10. `09_Error_Analysis.ipynb`
11. `10_Baseline_Comparison.ipynb`
12. `11_Ensemble_Model.ipynb` is experimental and should be treated as optional unless explicitly used for a new result table.

The helper module `notebooks_ethos/paths_config.py` creates local generated directories under `notebooks_ethos/data/`, `notebooks_ethos/models/`, and `notebooks_ethos/outputs/`.

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Alternatively, install from `pyproject.toml` with a compatible package manager.

## Data and model artifacts

The repository does not ship raw clinical data, generated corpora, or trained weights.

- MIMIC access must be handled through PhysioNet according to its credentialing and data use rules.
- Synthea data can be regenerated locally. The script `data/generate.sh` documents the expected local Synthea generation flow.
- Notebook outputs such as `corpus_train_*.txt`, `corpus_val_*.txt`, `medical_bert_mlm_*`, and `bert_triage_model_*` are generated locally and ignored by Git.

The lightweight figures and tables needed to inspect the manuscript results are retained under `notebooks_ethos/outputs/`.

## Manuscript

The final paper manuscript is included as `Banasik_Pater_Paper_2026.pdf`.

## Citation

Use `CITATION.cff` for citation metadata. The DOI/publication fields should be updated after the Springer publication is assigned.

## License

Code and repository materials are released under the MIT License unless a file states otherwise. Third-party datasets remain governed by their original licenses and access terms.
