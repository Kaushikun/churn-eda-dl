# Telco Customer Churn: EDA, Visualization and Deep Learning (MLP)

Case study for the Data Exploration and Visualization course (CO3 / CO4 / CO5).

**Problem:** predict and explain customer churn for a telecom company.
**Dataset:** IBM Telco Customer Churn, 7,043 customers, 21 fields
(source: https://github.com/IBM/telco-customer-churn-on-icp4d, file included in `data/`).
**Result:** churn is 26.5% and is driven by month-to-month contracts, short tenure, fiber-optic service,
electronic-check payment and lack of tech-support/security add-ons. A tuned MLP reaches test
ROC-AUC 0.845, recall 0.70 and F1 0.63 (classical baselines: AUC 0.849-0.851).

## Repository files

- README.md, requirements.txt
- Telco-Customer-Churn.csv: dataset
- churn_eda_dl.ipynb: executed notebook (EDA, visuals, model, evaluation)
- churn_eda_dl.py: same code as a script
- r1-r5 and v1-v13 .png files, interactive_scatter.html: all charts
- Project_Report.pdf / .docx: project report

## What is covered

* **Data cleaning:** hidden missing values in `TotalCharges` (11 new customers), duplicates, outliers (IQR), category consolidation.
* **Preprocessing:** encoding, standardization (fit on train only), stratified 70/15/15 split.
* **Visualization (7 stages):** 12 EDA figures, distribution fitting and Q-Q plots, binomial models, chi-square / Mann-Whitney tests, Cramer's V, mutual information, an interactive Plotly chart.
* **Model:** Keras MLP with batch-norm, dropout and L2, 12-configuration grid search, class weights, validation-chosen threshold.
* **Evaluation:** loss/accuracy curves, confusion matrix, ROC and PR curves, calibration, baseline comparison, permutation importance.

## How to run

Create folders `data/`, `notebooks/` and `figures/`, put the CSV in `data/` and the notebook in `notebooks/`, then run it.

Random seed is fixed to 42, so the reported numbers are reproducible.

## Author

\[kAUSHIKUN], \[23AM027], \[CSE(AIML)], \[IV YEAR]

