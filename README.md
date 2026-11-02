# Predicting 30-Day Hospital Readmission in Indian Hospitals

Machine learning and deep learning comparison (logistic regression, random forest, XGBoost, MLP) with a study-design audit,
on the public India Hospital Readmission Dataset (five linked tables, 120,000 admissions, 33 hospitals).

## Main findings (Google Colab run, seed 42)
- 112,587 admissions after excluding 7,413 in-hospital deaths; 12.6% readmitted within 30 days.
- Test ROC-AUC: Logistic Regression 0.756, MLP 0.754, XGBoost 0.752, Random Forest 0.748 (95% intervals overlap); prior-only baseline 0.500.
- Keeping in-hospital deaths raises ROC-AUC by about 0.02; random vs patient-grouped splitting changes it by less than 0.001.

## Repository layout
| Path | Content |
|---|---|
| `notebooks/01_preprocessing_colab.ipynb` | Data inspection, quality checks, merge, cleaning |
| `notebooks/02_modelling_colab.ipynb` | Models, evaluation, design audit, figures (run after 01) |
| `src/` | `preprocessing.py`, `models.py`, `visualization.py` used by the notebooks |
| `results/` | Colab outputs: `results_v2.json`, preprocessing report, figures |
| `paper/` | IEEE conference paper (`main.tex`, `refs.bib`, figures, `paper.pdf`) and TRIPOD+AI self-assessment |
| `data/` | Instructions for downloading the dataset (not included) |

## How to run
1. Download the dataset (see `data/README.md`).
2. Open each notebook in Google Colab (or Jupyter), set `DATA_DIR`, then Runtime > Run all (01 first, then 02). CPU is enough; about 10 minutes.
3. Local use: `pip install -r requirements.txt`.

## Paper
Compile `paper/main.tex` with any LaTeX distribution (class `IEEEtran`; BibTeX file `refs.bib`).
Author names and e-mail placeholders in the paper should be checked before submission.
