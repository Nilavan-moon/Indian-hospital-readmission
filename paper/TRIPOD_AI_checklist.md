# TRIPOD+AI self-assessment for the readmission paper (paper_v4)

This is a plain-language summary by topic, **not** the official numbered 27-item form. The official checklist (Collins et al., BMJ 2024;385:e078378,
doi 10.1136/bmj-2023-078378) should be downloaded from the TRIPOD+AI website and filled in item by item if the conference asks for it.
Status key: **Done** = reported in the paper, **Partial** = reported but incomplete, **Not done** = not addressed in this work.

| Topic | Status | Where / note |
|---|---|---|
| Title says it is a prediction-model study, with population and outcome | Done | Title |
| Structured abstract with data, methods, results, conclusion | Partial | One-paragraph abstract (conference format) |
| Background, rationale, existing models | Done | I Introduction, II Literature Survey |
| Objectives (develop vs evaluate, outcome, horizon) | Done | I Introduction (30-day readmission, discharge-time prediction) |
| Source of data and period | Partial | Public Kaggle dataset cited; label generation process not documented (stated as a limitation) |
| Participants, setting, eligibility | Done | III-2 (deaths excluded, with reason); 33 hospitals |
| Outcome definition | Partial | Supplied `readmitted_30d` flag used as given; not independently verified |
| Predictors, when measured | Done | III-3 (discharge-time inputs); Table II |
| Sample size justification | Not done | All 120,000 records used; no formal sample-size calculation |
| Missing data | Done | III-2: no missing cells; "None" insurance treated as a category |
| Data quality checks, duplicates, invalid values | Done | III-2 (from the preprocessing notebook) |
| Class imbalance handling | Done | Class weights; baseline; PR-AUC reported |
| Model type, hyperparameters, training | Done | III-4, Table III (hyperparameters, times) |
| Data splitting, no leakage across patients | Done | Patient-grouped 70/15/15; audit of the alternative |
| Performance measures: discrimination | Done | ROC-AUC with bootstrap 95% CI, PR-AUC, Table IV |
| Performance measures: classification at a threshold | Done | Accuracy, precision, recall, F1, confusion matrix |
| Performance measures: calibration | Partial | Brier score only; no calibration curve or recalibration |
| Comparison with a simple baseline | Done | Prior-only baseline and logistic regression |
| Internal validation | Partial | One held-out test set plus a temporal check (train before 2024, test on 2024); no repeated or cross-validation |
| External validation | Not done | Stated as future work |
| Model interpretation (important predictors) | Partial | Permutation importance; no SHAP |
| Fairness across subgroups (age, sex, insurance, region) | Not done | Subgroup readmission *rates* shown, not model performance by subgroup |
| Clinical usefulness (decision curve, intervention thresholds) | Not done | Mentioned as future work |
| Limitations | Done | V Result and Discussion (final paragraph) |
| Code and data availability | Partial | Notebooks provided to the team; dataset public; no public code repository yet |
| Ethics, patient involvement, funding, conflicts | Not done | Not stated; to be added by the authors if required |

**Summary:** the paper follows TRIPOD+AI for data, preprocessing, model and evaluation reporting, and states that external validation,
calibration curves, fairness analysis, sample-size justification, clinical-usefulness analysis and the ethics/funding items are not covered.
The paper text says "guided by" TRIPOD+AI, not "follows".
