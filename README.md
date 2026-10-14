# Intelligent Multi-Label Triage for Hematological Comorbidities using BERT

**Best Paper Award — Young AI Track, PP-RAI 2026** · Research code · Python / PyTorch / Transformers

An open research implementation accompanying our study of **multi-label triage for hematological comorbidities**. We combine synthetic and credentialed clinical data, pre-train a BERT-style masked language model on structured medical tokens, and fine-tune a multi-label classifier.

> **Research only.** This is not a clinical device or validated clinical decision-support system. Any real-world use would require independent clinical validation.

## Research at a glance

| Component | Approach |
| --- | --- |
| Research task | Predict multiple co-occurring hematological conditions rather than a single class |
| Data | Synthea-generated synthetic records and credentialed MIMIC data |
| Representation | Medical tokens derived from laboratory measurements and adaptive interview signals |
| Modeling | Masked language model pre-training, followed by multi-label fine-tuning |
| Evaluation | Baseline comparison, explainability, statistical analysis, and error analysis |
| Reproducibility | Ordered notebooks, code, lightweight figures and result tables; no restricted patient data |

The repository follows the research workflow used to produce the manuscript, not a deployable medical product.

## Publication and recognition

**Paper:** *Intelligent Multi-Label Triage for Hematological Comorbidities using BERT*  
**Authors:** Jan Banasik and Antoni Pater  
**Conference:** 7th Polish Conference on Artificial Intelligence (PP-RAI 2026)  
**Recognition:** Best Paper Award — Young AI Track

**Proceedings / DOI:** Publisher record and final citation will be linked here once officially confirmed. Until then, please do not cite this work as already published in Springer proceedings.

The research manuscript is a **separate scholarly work** and is **not licensed under this repository's MIT software license**. Distribution of any manuscript version is subject to the applicable author–publisher agreement and self-archiving rules.

For the implementation, GitHub's **Cite this repository** feature reads [`CITATION.cff`](CITATION.cff).

## Explore the research artifacts

- **[Figures](notebooks/outputs/plots/)** — visualizations retained for inspecting research results.
- **[Result tables](notebooks/outputs/tables/)** — lightweight evaluation outputs retained in Git.
- **[Notebooks](notebooks/)** — the experimental pipeline and analysis.
- **[Citation metadata](CITATION.cff)** — software citation; publication metadata will be updated after verification.

The numerical results should be read in the context of the reported data splits, baselines and study limitations; this README does not substitute for the full research methodology.

## Reproduce the workflow

Python **3.10+** is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks from the `notebooks/` directory in the following order:

| Stage | Notebook |
| --- | --- |
| 00 · Methodology | `00_Methodology.ipynb` |
| 01 · Synthetic data | `01_Preprocessing_Synthea.ipynb` |
| 02 · Clinical data | `02_Preprocessing_MIMIC.ipynb` |
| 03 · Merge | `03_Merge_Datasets.ipynb` |
| 04 · Pre-training | `04_Train_MLM_Model.ipynb` |
| 05 · Fine-tuning | `05_Fine_Tune_Model.ipynb` |
| 06 · Explainability | `06_Explainability.ipynb` |
| 07 · Evaluation | `07_Advanced_Evaluation.ipynb` |
| 08 · Statistics | `08_Statistical_Analysis.ipynb` |
| 09 · Error analysis | `09_Error_Analysis.ipynb` |
| 10 · Baselines | `10_Baseline_Comparison.ipynb` |

`11_Ensemble_Model.ipynb` is **experimental and optional**; it should not be taken as part of the core reported results without further verification.

The helper `notebooks/paths_config.py` initializes local generated directories under `notebooks/data/`, `notebooks/models/` and `notebooks/outputs/`.

## Data access and responsible use

Raw clinical data, generated corpora, trained models, checkpoints and MLflow state **are not distributed** by this repository.

- **MIMIC:** Obtain access independently through [PhysioNet](https://physionet.org/) and comply with the applicable data-use agreement. Do not commit or redistribute credentialed records.
- **Synthea:** Generate synthetic records locally; see [`data/generate.sh`](data/generate.sh) for the expected workflow.
- **Outputs:** The figures and tables committed here are lightweight research artifacts; recreating the complete experiments requires the appropriate data access and compute environment.

## Authors

- [Jan Banasik](https://github.com/JanBanasik)
- [Antoni Pater](https://github.com/antoniopater)

## License and publication rights

The repository's [MIT License](LICENSE) applies to the **source code and associated software documentation**. It does **not** license the manuscript PDF, Springer-published material, restricted datasets, or third-party content. These materials remain subject to their respective rights and terms.

The code and research outputs are provided **as-is for academic investigation** and are not validated for clinical use.
