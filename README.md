# Preoperative CI–SVRI phenotypes and postoperative AKI

Statistical analysis code supporting the CI–SVRI phenotype study in older patients undergoing elective off-pump CABG. This repository contains code only; participant data, manuscript files and internal audit records are not included.

## Run

Python 3.12 was used for the study. Install dependencies with `python -m pip install -r requirements.txt`.

```sh
python analysis.py --cohort /path/to/authorised_analysis_cohort.csv --output results
python analysis.py --cohort /path/to/authorised_analysis_cohort.csv --output results --section all
```

The default command fits the main Firth logistic model and two reference-range sensitivity models. `--section all` adds the nine reported post hoc subgroup variables, four-knot restricted cubic splines, the incremental-fit test and univariable/multivariable coefficient table. No single-CI-change outcome model is included.

## Input contract

An analysis-ready CSV must contain one row per participant (the adopted study cohort has 1,949 participants and 209 AKI events). Required columns are `AKI_7d`, `Cardiac_index_baseline`, `preop_SVRI`, `trial_arm`, `Age`, `Male`, `BMI`, `Smoking`, `History_of_myocardial_infarction`, `Arrhythmia`, `Diabetes`, `Hypertension`, `eGFR`, `Hb`, and `Left_ventricular_EF`.

CI is in L/min/m²; SVRI in dyn·s·cm⁻⁵·m²; age in years; BMI in kg/m²; eGFR in mL/min/1.73 m²; Hb in g/L; LVEF in %. Binary covariates use 0/1, with male=1, current/former smoking=1, condition present=1, and trial intervention=1. AKI_7d is the author-confirmed creatinine-based outcome within seven postoperative days. eGFR is the author-confirmed 2021 CKD-EPI value. This repository starts from those prepared inputs; it does not implement access to source hospital records or outcome adjudication.

The primary normal ranges are CI 2.5–4.0 and SVRI 2000–2400 (inclusive). Sensitivity models use CI lower bound 2.2 or SVRI limits 1970–2390. Subgroups use age 65, BMI 24, eGFR 60 and LVEF 50; other subgroup variables are binary. Continuous age remains in the age-interaction model. Firth estimates use profile-likelihood intervals and constrained full-model penalised likelihood-ratio tests. Standardised risks use the delta method. Interaction tests are exploratory and not multiplicity-adjusted.

## Scope and provenance

Estimation functions were extracted without changing their calculations from the adopted analysis implementation. The entry point was adapted to accept a user-supplied input path and an isolated output directory, removing local paths, historical result-file dependencies and participant-specific audit records. The two non-reported smoking/Hb subgroup explorations are not included. Source-function fingerprints are listed in `provenance.json`.

This is an analysis-code release, not a complete raw-data-to-manuscript reproduction package. Intraoperative waveform extraction and final manuscript artwork are outside this release. Access to deidentified study data remains by reasonable request to the corresponding authors and is separate from public code access. No data-access permission is conferred by this repository.
