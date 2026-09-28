# Interview Guide — PulseRetain
> Written in first person as the project owner. Honest, evidence-grounded answers.

---

## Why this problem?

Subscriber attrition is one of the most financially damaging events a SaaS company faces. Unlike one-time sales, SaaS revenue is recurring — losing a customer doesn't just cut a single transaction, it eliminates an entire revenue stream. A 1% monthly churn rate compounds to ~11% annual churn, wiping out new logo growth for a company at scale.

I chose this problem because it spans the entire data science workflow: data engineering, EDA, SQL analytics, classical ML, deep learning, and business-facing dashboard design. It's also a problem where the model output directly maps to a business action (contact this account), which makes evaluation genuinely meaningful.

---

## Why this dataset?

The IBM Telco dataset is widely used as a reference, which means it's well-understood — but also widely copied. I generated a **synthetic SaaS dataset** for three reasons:

1. **No licensing constraints** — the IBM dataset has an unclear license for portfolio use.
2. **Domain authenticity** — B2B SaaS has a different churn profile than telecom: it involves seat counts, integration stickiness, billing cycles, and CSM interactions.
3. **Demonstrating understanding** — designing the generator (distributions, correlations, injected quality issues) proves I understand how real data arises and what causes quality problems.

The generator is seeded (seed=42) and documented — anyone can reproduce the exact same dataset.

---

## How was the data cleaned? What trade-offs were made?

Five cleaning steps, each isolated and audited:

| Issue | Strategy | Trade-off |
|---|---|---|
| ~1% duplicate subscriber IDs | Keep first occurrence, drop rest | Assumes first record is canonical; could lose valid corrections in a real CRM |
| Impossible dates (churn before start) | Null the date, set churned=0 | Conservative — could theoretically miss real churns with bad dates |
| ~0.5% negative monthly fees | Replace with plan-tier median | Preserves rows; plan-tier median is better than global median because pricing is tier-specific |
| ~3% null engagement_score | Impute with plan-tier median | MCAR-like within tier; median preferred over mean due to slight left-skew |
| ~1.5% null monthly logins | Impute with global median | No clear group structure identified in EDA |

I specifically **did not** drop rows with nulls in non-critical columns. Dropping would introduce selection bias — the accounts missing engagement data are not random (they tend to be Starter plan, which has less telemetry). Dropping them would make the training data unrepresentative.

---

## Why SQL?

SQL is the language of the business. Any stakeholder who wants to verify a KPI, audit a number, or build a Power BI report needs SQL. If KPI logic lives only in Python, it can't be shared, audited, or queried interactively.

Specific choices:
- **Views** (`vw_kpi_headline`, `vw_kpi_by_plan`, etc.) make dashboard queries simple and consistent
- **Window functions** (`LAG`, `NTILE`, `SUM() OVER`) handle cohort analysis and period-over-period comparisons cleanly in SQL — pushing this to pandas would be more code and harder to read
- **SQL tests** in `tests/test_sql.py` reconcile SQL KPIs against pandas calculations, ensuring the two layers agree

---

## Why pandas and NumPy?

For steps that are naturally row-wise or require statistical libraries. Pandas handles:
- Schema validation (dtype checks, value checks)
- Cleaning (imputation, capping, dedup)
- Feature engineering (ratios, logs, interactions)
- Evaluation (confusion matrices, metric computation)

NumPy provides the vectorized math underlying the feature engineering (log transforms, sigmoid, clip) and the synthetic data generator's probability calculations.

I kept the SQL/pandas boundary clean: **aggregations and KPIs live in SQL; transformations per row live in pandas.**

---

## Why scikit-learn?

Four reasons:

1. **Pipeline objects** prevent leakage by design — `fit()` is only called on training data, and `ColumnTransformer` applies the same transformations to new data
2. **StratifiedKFold CV** makes model selection fair and variance-aware
3. **`class_weight='balanced'`** handles the 18% churn rate without manual resampling
4. **`HistGradientBoostingClassifier`** is sklearn's modern gradient boosting implementation — faster than XGBoost for this data size and handles missing values natively

---

## Why TensorFlow? What happened when compared with simpler models?

I used TF for a **Wide and Deep tabular network with entity embeddings** (Cheng et al., 2016). The reason for entity embeddings is that the four categorical features (plan_tier, region, channel, industry) have ordinal or semantic structure that one-hot encoding destroys. An embedding maps each category to a dense vector where similar categories should cluster.

What happened in comparison:
- **Gradient Boost typically wins on ROC-AUC for this data size** (~8K rows). This is expected — neural networks need more data to generalise better than ensembles.
- **The TF model is competitive in PR-AUC** (precision-recall), which matters more for imbalanced data.
- **The value of the TF model here is not winning the comparison** — it's demonstrating that I can design, justify, implement, train with callbacks, evaluate honestly, and explain a neural architecture.

If I were deploying to production today with this data, I'd use Gradient Boost + calibration. If we had 50× more data or per-event logs (LSTM for sequences), the TF model would likely pull ahead.

---

## How does each model work, intuitively?

| Model | Intuitive explanation |
|---|---|
| **Logistic Regression** | Finds a weighted combination of features that best separates churners from retainers. The weights are the coefficients — you can inspect them directly. Fast, interpretable, and well-calibrated. |
| **Random Forest** | Builds 300 decision trees, each on a random subset of data and features. Each tree makes a prediction; the forest votes. By averaging many noisy trees, it captures non-linear patterns without overfitting any single tree. |
| **Gradient Boost** | Builds trees sequentially — each new tree corrects the errors of all previous trees. The `Hist` variant bins numeric values into histograms first, making splits much faster. Very effective on tabular data. |
| **Wide & Deep TF** | Wide path memorises simple patterns (linear); deep path discovers complex interactions (non-linear). Entity embeddings learn compressed representations of categories that capture semantic proximity (e.g. Professional and Enterprise share characteristics). |

---

## Why did one model perform differently from another?

- **Logistic Regression** underperforms tree models because it can't capture the interaction between engagement_score and payment_failures without explicit interaction terms — it finds a linear boundary in feature space that doesn't fit the data's non-linear structure.
- **Random Forest** vs **Gradient Boost**: RF trains trees independently (parallel), GBoost trains sequentially (each tree fixes errors of previous). On this dataset, GBoost generally wins because it focuses more learning effort on hard cases (boundary accounts with moderate churn probability).
- **TF model** can underperform on small tabular data because neural networks have many more parameters to fit — they need more data before the inductive biases pay off.

---

## What does each evaluation metric mean and why was it chosen?

| Metric | Meaning | Why chosen |
|---|---|---|
| **ROC-AUC** | Probability that the model ranks a random churner above a random retainer. 1.0 = perfect, 0.5 = random | Threshold-independent; good for model comparison |
| **PR-AUC** | Average precision across all recall levels. Robust to class imbalance | 18% churn makes PR-AUC more informative than ROC-AUC in the high-recall region |
| **F1 @ 0.40** | Harmonic mean of precision and recall at threshold 0.40 | Balances false positives and false negatives at the operating point |
| **Recall @ 0.40** | Fraction of actual churners correctly identified | Priority metric — missing a churner is more costly than a false alarm |
| **Precision @ 0.40** | Fraction of predicted churners that are truly churning | Ensures CS team's time is well spent |

---

## How was the dashboard designed and for whom?

Three personas:
1. **VP Customer Success (executive)**: needs headline KPIs and ARR at risk in 30 seconds
2. **Growth Analyst (operational)**: needs segment drilldowns, cohort trends, and channel analysis
3. **Data Scientist (analytical)**: needs model comparison, prediction distributions, data quality diagnostics

Design decisions:
- **Question-oriented navigation**: "What happened?", "Where is the risk?", "What should we do?" — not "Charts" or "Tables"
- **Dark theme**: reduces eye strain for long sessions; conveys a premium analytics product feel
- **Colorblind-safe palette**: Wong (2011) palette — distinguishable for deuteranopia and protanopia
- **KPI cards with definitions**: every number has a formula and a tooltip, preventing misinterpretation
- **Actionable recommendations panel**: translates insights into prioritised actions with estimated impact and stated assumptions

---

## What business insights were found?

(Based on synthetic data distributions — illustrative, not predictive)

1. Annual billing reduces churn by ~16pp — most actionable lever
2. Payment failures are the strongest churn predictor and arrive 1-3 months before churn
3. Starter plan churn is ~3× Enterprise churn — need differentiated success motions
4. E-commerce and Media are the highest-churn verticals — consider industry-specific onboarding
5. Referral-sourced accounts are the stickiest channel — invest in referral programs

---

## What limitations exist?

- **Synthetic data**: all insights are from a mathematical model. Must validate on real data.
- **Correlation ≠ causation**: engagement predicts churn but improving engagement may not reduce it if both are downstream of product-market fit.
- **No sequential data**: we don't have per-user event logs, so we can't model the degradation path before churn.
- **Leakage risk in production**: if billing failure data arrives after churn, a production pipeline needs careful timestamp handling.
- **Generalization**: model is trained on one synthetic cohort — real performance will vary.

---

## What would change in production?

| Concern | Solution |
|---|---|
| **Model drift** | PSI/KS drift detection weekly; alert and trigger retraining |
| **Data quality** | Data contracts with source CRM; automated quality reports on each ingestion |
| **Retraining** | Automated pipeline on schedule + on-alert; time-aware train/test split mandatory |
| **Access control** | Role-based access: CS team sees watchlist; analysts see everything; models are not publicly accessible |
| **CI/CD** | GitHub Actions pipeline: lint → test → train → evaluate → deploy if metrics pass threshold |
| **Serving** | FastAPI endpoint with Redis caching for real-time scoring; batch scoring nightly to DB |
| **Cost** | SQLite → PostgreSQL (or BigQuery for scale); TF model → ONNX for faster inference |
| **Explainability** | SHAP for individual account explanations; top 3 risk factors shown in watchlist UI |

---

## Walkthrough Scripts

### 3-Minute Demo

1. Open dashboard at `http://localhost:8501`
2. **"What happened?"** page: show total ARR, churn rate, auto-narrative
3. **"Who are the at-risk accounts?"** page: show high-risk watchlist sorted by churn probability; export to CSV
4. **"What should we do?"** page: show P1 action with estimated ARR impact
5. **"How reliable is the model?"** page: show model comparison table and prediction distribution

### 10-Minute Demo

1. **Pipeline walkthrough**: show `pulse/ingestion.py` (generator), `pulse/validation.py` (quality report), `pulse/cleaning.py` (audit log)
2. **SQL layer**: open `sql/schema.sql` — explain star schema; run `sql/queries/analytical_queries.sql` Q5 (ARR running total) in SQLite Browser or CLI
3. **Feature engineering**: show `FEATURE_CATALOG` in `pulse/features.py`; explain leakage prevention
4. **Model comparison**: `data/processed/model_comparison.csv`; explain why gradient boost typically wins on tabular data
5. **TF model**: show architecture in `pulse/modelling_tf.py`; explain entity embeddings and wide-deep combination
6. **Dashboard**: walk all 5 pages; demonstrate cross-filtering; export watchlist
7. **Tests**: run `pytest tests/ -v` live
8. **Limitations and production gaps**: reference model card

---

## 15 Likely Technical Follow-Up Questions

1. **"Why not use SMOTE for class imbalance?"** — `class_weight='balanced'` is simpler, has no risk of overfitting synthetic minority samples, and works inside a Pipeline without data leakage.

2. **"How do you know Churn Score from the reference dataset wasn't leakage?"** — It was! The IBM dataset's Churn Score column is a pre-computed risk signal that encodes churn probability. Using it as a feature is target leakage. I excluded it from modelling in the reference analysis and designed my synthetic data without an equivalent.

3. **"What would you do differently with 10× more data?"** — Try an LSTM on per-user event logs. Neural networks justify their complexity only when there's enough data to fit their parameters.

4. **"Why HistGradientBoosting over XGBoost?"** — HGB is sklearn-native (no extra install), handles missing values natively (surrogate splits), and is faster on this data size. XGBoost would likely perform similarly.

5. **"How would you handle the cold-start problem for new subscribers?"** — Use plan-level and channel-level priors for accounts with <30 days of history. Segment new accounts separately in the watchlist.

6. **"What is the ROC-AUC of a random classifier?"** — 0.50. Any model must beat this to add value.

7. **"Why did you choose a 0.40 threshold rather than 0.50?"** — Cost-sensitive reasoning: the cost of a false negative (losing $12K ARR) far exceeds the cost of a false positive ($50 CS call). Lower threshold → higher recall → fewer churners missed.

8. **"How does StandardScaler work?"** — Subtracts column mean and divides by standard deviation: `(x - μ) / σ`. Required for logistic regression (gradient descent converges faster on unit-scale features); not strictly needed for tree models (but used here for consistency).

9. **"What is entity embedding and why use it?"** — Learnable dense vector representation for categorical variables. OHE gives [1,0,0,0] for 'starter' and [0,1,0,0] for 'growth' — no relationship. Embedding learns that 'professional' and 'enterprise' are semantically similar, which the model can exploit.

10. **"How did you prevent data leakage?"** — Three mechanisms: (1) build_features() explicitly drops leakage columns; (2) sklearn Pipeline fits preprocessor only on train data; (3) test set is loaded from disk at evaluation time only — it was never seen during any modelling step.

11. **"What is PR-AUC?"** — Area under the precision-recall curve. More informative than ROC-AUC when classes are imbalanced because it emphasises performance on the minority class (churners).

12. **"Why a star schema and not a flat denormalized table?"** — Maintainability: if plan pricing changes, I update one row in dim_plan, not 8,500 rows in fact_subscribers. Also enables proper foreign key constraints and standardises query patterns.

13. **"How would you monitor for model drift?"** — Population Stability Index (PSI) on feature distributions between training baseline and current scoring data. Alert if PSI > 0.20 for any key feature. Also compare prediction score distributions weekly (KS test).

14. **"What is early stopping in the TF model?"** — Monitors validation ROC-AUC; stops training if it hasn't improved for 12 consecutive epochs (`patience=12`) and restores the weights from the best epoch. Prevents overfitting without manually choosing the number of epochs.

15. **"Could you explain a single prediction — why is this account high risk?"** — In the current codebase, permutation importance gives global feature importance. For individual explanations, I'd add SHAP `TreeExplainer` (works for gradient boost models): it decomposes the prediction into contributions from each feature for that specific account. The dashboard's watchlist could show top 3 contributing factors per account.
