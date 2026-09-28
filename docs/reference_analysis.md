# Reference Analysis — Private Note
> **For the candidate's internal understanding only. Not for public distribution.**

## 1. Problem Statement and Target Users
The reference project predicts whether a telecom customer will cancel ("churn").
Target users: business analysts / ops managers at a telecom company.
No persona distinction between executive / ops / analyst in the UI.

## 2. Directory Structure
Single monolithic notebook, one Power BI .pbix, no pipeline separation, no SQL, no tests.

## 3. Data — 7,043 rows × 33 columns (IBM Telco xlsx)
Key groups: Identity, Demographics, Services, Contract, Financials, Targets (Churn Label/Value/Score, CLTV, Churn Reason).

## 4. Dependencies
Inferred: pandas, numpy, sklearn, tensorflow, xgboost. No requirements.txt.

## 5. ML Components
Random Forest, XGBoost, ANN on binary churn. Single train/test split, no CV, no calibration.
Feature engineering: label encoding only. No leakage analysis. Churn Score / CLTV may leak.

## 6. Dashboard (Power BI only)
KPI cards: Total/Active/Churned customers, Churn Rate, Revenue, High-Risk count.
Charts: churn by contract, internet, gender, senior. Risk table. Model comparison bars.

## 7. Weaknesses
1. No reproducible pipeline (all inline in notebook).
2. No schema validation or data-quality reporting.
3. No SQL layer — all aggregation in pandas.
4. No separation of concerns.
5. Weak evaluation — accuracy-focused, single split, no cost-sensitivity.
6. No leakage check for Churn Score / CLTV columns.
7. No tests, config, or logging.
8. Power BI only — requires proprietary software.
9. No model card, interview guide, or data dictionary.
10. Accessibility not considered.
11. No actionable recommendations panel.
