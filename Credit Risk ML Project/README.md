# Consumer Credit Default Risk Analysis & Underwriting Optimization

An end-to-end machine learning and financial decision-science pipeline for predicting 2-year serious credit delinquency (`SeriousDlqin2yrs`) using the **Give Me Some Credit** dataset ($N = 150,000$ borrower records; $148,366$ after cleaning).

Beyond standard classification benchmarking, this project emphasizes **probability calibration** and **cost-sensitive underwriting optimization**—translating calibrated Probability of Default (PD) estimates into an automated credit approval policy that maximizes expected portfolio profit.

---

## Table of Contents
1. [Business Problem & Objectives](#1-business-problem--objectives)
2. [Dataset & Data Hygiene Audit](#2-dataset--data-hygiene-audit)
3. [Exploratory Data Analysis & Key Risk Drivers](#3-exploratory-data-analysis--key-risk-drivers)
4. [Feature Engineering & Preprocessing](#4-feature-engineering--preprocessing)
5. [Modeling & Validation Protocol](#5-modeling--validation-protocol)
6. [Results & Probability Calibration](#6-results--probability-calibration)
7. [Financial Impact & Underwriting Policy](#7-financial-impact--underwriting-policy)
8. [Limitations & Production Roadmap](#8-limitations--production-roadmap)
9. [Repository Structure & Reproducibility](#9-repository-structure--reproducibility)

---

## 1. Business Problem & Objectives
* **Objective:** Predict whether an applicant will experience serious delinquency (90+ days past due or worse) within the next 24 months (`SeriousDlqin2yrs = 1`) and output well-calibrated default probabilities ($\hat{p}_i \in [0, 1]$).
* **Stakeholder Decision:** Consumer credit underwriting teams setting automated PD cutoffs to maximize net portfolio lending profit while reducing the approved portfolio bad rate from the **~6.7% base rate** (6.68% in the raw file, 6.73% after cleaning) to **< 3.0%**.
* **Evaluation Metrics:**
  * **Primary Selection Metric:** Stratified Cross-Validation **ROC-AUC** (pairwise discrimination / rank-ordering power): 5-fold for the logistic models and GBM, 3-fold for the EBM and SVM (see Section 5).
  * **Minority-Class Ranking:** **PR-AUC (Average Precision)** to evaluate precision-recall trade-offs under class imbalance (~6.7% positive class).
  * **Probabilistic Truthfulness:** **Brier Score** and **Reliability Diagrams (Calibration Curves)** to check that predicted scores reflect true empirical default frequencies suitable for expected-loss pricing. The reliability diagram is the direct calibration check; Brier score combines calibration and discrimination.

---

## 2. Dataset & Data Hygiene Audit
The raw training dataset (`cs-training.csv`) contains **150,000 rows** and **11 variables** (1 binary target + 10 borrower features). Our initial audit uncovered several data quality anomalies and reporting artifacts that would severely distort linear coefficients and tree splits if left untreated:

| Anomaly / Issue | Audit Finding | Pipeline Cleaning Action | Rationale |
| :--- | :--- | :--- | :--- |
| **Sentinel Codes (`96`, `98`) in Past-Due Columns** | Exactly `269` rows share codes `96`/`98` across all three delinquency windows (54.65% default rate; perfect collinearity). | Consolidate into a single binary indicator `PastDue_SpecialCode`, then replace `96`/`98` with `NaN`. | Preserves the high-risk signal (~8x base rate) without corrupting numeric delinquency counts or adding 3 collinear flags. |
| **Missing & Zero `MonthlyIncome`** | `29,731` missing (`19.82%`) + `1,634` zero-income entries. | Convert `0` to `NaN`, create a binary `IncomeMissing` flag, and median-impute inside the CV `Pipeline`. | Captures non-reporting behavior while preventing test-set data leakage during imputation. |
| **Extreme `DebtRatio` Outliers** | `29,646` rows above 5, with values up to `329,664` (caused by missing/zero income denominators where raw dollar debt was recorded instead of a ratio). | Replace `DebtRatio > 5` with `NaN` and median-impute within the CV `Pipeline`. | Removes unscaled dollar-debt denominator artifacts. |
| **Invalid Borrower Age** | `1` record with `age = 0`. | Set `0` to `NaN` and drop row (`dropna(subset=['age'])`). | Data entry error; dropping 1 row has zero statistical impact. |
| **Duplicate Records** | `609` exact duplicates in the raw file; `1,633` duplicate borrower profiles after sentinel/outlier masking. | Remove via `drop_duplicates()` (applied after masking), leaving **148,366** clean records. | Prevents identical profiles from leaking across train/test splits. The extra ~1,024 rows are profiles that differ only in values later blanked out, so deduplicating before masking is the stricter alternative. |
| **Missing `NumberOfDependents`** | `3,924` missing records (`2.62%`). | Retain as `NaN` and median-impute inside the CV `Pipeline`. | Low missingness; in-pipeline imputation prevents leakage. |

---

## 3. Exploratory Data Analysis & Key Risk Drivers
1. **Severity-Ordered Delinquency Signal:**
   * Borrowers with zero past-due events across all three windows have a baseline default rate of **2.75%**.
   * A single delinquency event increases empirical default risk sharply by severity window:
     * **30–59 Days Late (`0 -> 1`):** Default rate jumps by **~11 percentage points** (`4.0%` $\rightarrow$ `15.0%`).
     * **60–89 Days Late (`0 -> 1`):** Default rate jumps by **~26 percentage points** (`5.1%` $\rightarrow$ `31.0%`).
     * **90+ Days Late (`0 -> 1`):** Default rate jumps by **~29 percentage points** (`4.7%` $\rightarrow$ `33.7%`).
2. **Revolving Credit Exhaustion:**
   * Default rates rise non-linearly with credit utilization: borrowers with exactly **0%** utilization default at **3.20%**, those utilizing **>0–25%** of their unsecured credit lines default at just **2.02%**, whereas those at **75–100%** default at **18.63%** (~2.8x the base rate), and borrowers **over 100% utilization** default at **37.25%** (~5.5x).
3. **Lifecycle & Financial Cushion:**
   * Default risk declines monotonically with age (from **11.78%** for borrowers under 30 down to **2.29%** for borrowers aged 70+) and with monthly income (**9.46%** for `< $3k/mo` down to **4.32%** for `$10k+/mo`).

---

## 4. Feature Engineering & Preprocessing
* **Zero-Preserving Log Transforms (`np.log1p`):** `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines` have many legitimate `0.0` values among low-risk borrowers, and `MonthlyIncome` is heavily right-skewed (its zeros were already converted to `NaN` as a reporting artifact). Applying $\log(1 + x)$ to all three compresses extreme right tails while mapping `0.0` $\rightarrow$ `0.0` (avoiding the pitfall of $\log(x)$, which converts zeros into `NaN` and imputes low-risk borrowers to median risk).
* **Leakage-Free Outlier Capping:** `LogRevolvingUtilization` is capped at the 99th percentile computed strictly on `X_train` and applied to both `X_train` and `X_test`.
* **Collinearity Pruning:** Individual delinquency windows (`30-59`, `60-89`, `90+`) are retained rather than `TotalPastDueEvents` so models can weight severity differences distinctly. A household-income feature (`LogIncome2Members` = log1p of income per household member) was explored in EDA but excluded to avoid collinearity with `LogMonthlyIncome` and `NumberOfDependents`.

**Final 12-Feature Modeling Subset (`SELECTED_FEATURES`):**
`LogRevolvingUtilization`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `PastDue_SpecialCode`, `LogDebtRatio`, `LogMonthlyIncome`, `IncomeMissing`, `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines`, `NumberOfDependents`.

---

## 5. Modeling & Validation Protocol
* **Train / Holdout Split:** 80% Training (`118,692` rows) / 20% Holdout Test (`29,674` rows), stratified on `SeriousDlqin2yrs`.
* **Calibration Preservation:** Models are trained **without** `class_weight="balanced"`. Artificial 50/50 class reweighting skews predicted probabilities away from the ~6.7% empirical prior and degrades Brier calibration.
* **Cross-Validation Folds:** The logistic models and GBM are tuned with **5-fold** Stratified CV on `X_train`. The EBM and the SVM use **3-fold** CV for compute reasons (the SVM on a 15k subsample), so their CV scores are not strictly like-for-like with the others.
* **Model Suite:**
  * Baseline L2 Logistic Regression (`C=1.0`)
  * Tuned Ridge (`L2`), Lasso (`L1`), and Elastic Net Logistic Regression
  * Tuned Gradient Boosting Classifier (`GBM`)
  * Tuned Explainable Boosting Machine (`EBM` / Glassbox GAM)
  * Subsampled RBF Support Vector Machine (`15,000` stratified training subsample)

---

## 6. Results & Probability Calibration

Models were selected by out-of-fold Cross-Validation ROC-AUC on `X_train` (5-fold for GBM and the logistic models, 3-fold for EBM and SVM) before evaluating on the held-out test set (`X_test`):

| Rank | Model | Best CV ROC-AUC | Holdout Test ROC-AUC | Test PR-AUC (AP) | Test Brier Score | Best Hyperparameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Gradient Boosting (`gbm`)** | **0.8636** | **0.8677** | **0.4075** | **0.04877** | `lr=0.05, max_depth=4, n_estimators=200` |
| **2** | **Explainable Boosting (`ebm`)** | 0.8618 | 0.8666 | 0.4049 | 0.04902 | `interactions=5, max_rounds=300, lr=0.02` |
| **3** | **Ridge Logistic Regression** | 0.8486 | 0.8565 | 0.3954 | 0.05020 | `C=0.01 (L2)` |
| **4** | **Elastic Net Logistic Regression** | 0.8486 | 0.8565 | 0.3953 | 0.05020 | `C=0.01, l1_ratio=0.2` |
| **5** | **Lasso Logistic Regression** | 0.8483 | 0.8563 | 0.3954 | 0.05020 | `C=0.01 (L1)` |
| **6** | **Baseline Logistic Regression** | 0.8483 | 0.8563 | 0.3950 | 0.05021 | `Fixed (C=1.0, L2)` |
| **7** | **RBF SVM (15k Subsample)** | 0.6796 | 0.6774 | 0.2970 | 0.05745 | `C=0.5, gamma='scale'` |

* **Champion Model:** **Gradient Boosting (`gbm`)** achieved the highest CV ROC-AUC (**0.8636**) and Test ROC-AUC (**0.8677**), outperforming the linear log-odds models (~0.849 CV) consistent with non-linear threshold effects in revolving utilization and delinquency counts. The GBM–EBM gaps (0.0018 CV, 0.0011 test) are small, so the two are effectively tied and `gbm` is selected by the stated rule.
* **RBF SVM:** At ~0.68 ROC-AUC the SVM falls far below even the linear baseline and has an erratic reliability curve. This is likely a limitation of the 15k subsample and outlier-heavy count features under an RBF kernel (untested), so treat it as a weak benchmark rather than evidence about kernel methods.
* **Calibration Verification:** `gbm` and `ebm` achieved Brier scores of **~0.0488 / 0.0490**—about a **22% reduction in mean squared probability error** compared to a naive base-rate forecast ($BS_{\text{naive}} = \bar{y}(1-\bar{y}) \approx 0.0628$ at the 6.73% test base rate)—and closely tracked the $45^\circ$ diagonal on the reliability diagram. The logistic models also track the diagonal, with slight under-prediction in the top decile.

---

## 7. Financial Impact & Underwriting Policy

To translate model probabilities into an actionable lending policy, we define loan-level unit economics over the 2-year horizon:
* **Approved Good Loan (`True Negative`):** **$+\$1,200$** net interest & fee profit.
* **Approved Defaulted Loan (`False Negative`):** **$-\$10,000$** net Loss Given Default ($\text{LGD} \times \text{EAD}$).
* **Declined Application (`True Positive` / `False Positive`):** **$\$0$** direct cash flow.

### Theoretical Break-Even Probability Threshold ($p^*$)
Setting expected profit $\mathbb{E}[\Pi \mid \text{Approve}] = (1 - p^*)(\$1,200) + p^*(-\$10,000) = 0$ yields the break-even default probability:

$$p^* = \frac{\$1,200}{\$1,200 + \$10,000} = \frac{1,200}{11,200} \approx 10.71\%$$

### Holdout Test Set Portfolio Simulation ($N = 29,674$)

The headline policy uses the theoretical cutoff, which does not rely on test labels to choose the threshold.

| Underwriting Strategy | PD Cutoff | Approval Rate | Approved Good Loans | Approved Bad Loans | Portfolio Bad Rate | Total Net Profit | Profit / Applicant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Model Baseline (Approve All)** | $100\%$ | $100.0\%$ | $27,677$ | $1,997$ | $6.73\%$ | $\$13,242,400$ | $\$446.26$ |
| **GBM Cost-Optimal Policy** | **$\text{PD} < 10.71\%$** | **$84.8\%$** | **$24,520$** | **$658$** | **$2.61\%$** | **$\$22,844,000$** | **$\$769.83$** |
| **Net Impact ($\Delta$)** | — | $-15.2\%$ | $-3,157$ | **$-1,339$** | **$-4.12\text{ pp}$** | **$+\$9,601,600$** | **$+\$323.57$** |

By declining the riskiest **15.2%** of applicants ($\text{PD} \ge 10.71\%$; 4,496 applicants), the GBM underwriting policy prevents **1,339 defaults** (saving **$\$13.39\text{M}$** in credit losses) at the cost of foregoing **3,157 good loans** ($\$3.79\text{M}$ in foregone interest), generating **+$9.60M in net incremental profit** on the test sample.

**Sensitivity check:** sweeping cutoffs from 1% to 50% and picking the profit-maximizing one on the test set (which is optimistic, since it uses test labels) yields $\$22,853,200$ (84.0% approval, 2.52% bad rate, +$\$9,610,800$ vs. baseline), only **$\$9,200$ (~0.04%)** more than the theoretical cutoff.

---

## 8. Limitations & Production Roadmap
* **Selection Bias (Reject Inference):** Outcomes are only observed for applicants historically granted credit. Deploying into production requires reject-inference techniques or parceling on bureau performance of declined cohorts.
* **Static Cross-Sectional Snapshot:** The dataset lacks loan origination timestamps and macroeconomic covariates (e.g., unemployment rate, interest rate regime). Production validation requires **Out-of-Time (OOT)** backtesting across origination vintages.
* **Evaluation Caveats:** All results come from a single stratified holdout split; the EBM and SVM were tuned with 3-fold CV (SVM on a 15k subsample); and the profit sweep's optimum is selected on the test set, so it is a sensitivity check only.
* **Bureau Trended Data:** Incorporating 6-to-24 month credit utilization velocity (distinguishing borrowers paying down debt from those rapidly exhausting credit lines) would substantially sharpen separation.
* **Fair Lending & Monotonicity Constraints:** Under ECOA / Regulation B, `age` cannot be used to penalize elderly applicants, and adverse-action reason codes require monotonic risk relationships. Enforcing **monotonicity constraints** in LightGBM/XGBoost/EBM guarantees that predicted PD never decreases when utilization or delinquency counts increase.

---

## 9. Repository Structure & Reproducibility

```text
├── datasets/
│   └── cs-training.csv          # Give Me Some Credit training dataset (150,000 rows)
├── pickles/                     # Auto-generated workspace checkpoints (.pkl)
├── credit_risk.ipynb            # Main analysis, EDA, modeling, and business impact notebook
└── README.md                    # Project documentation
```

### Setup Instructions

1. Clone the repository and install core dependencies:

   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tqdm interpret jupyter kaggle
   ```

2. Download `cs-training.csv` via the Kaggle CLI (`kaggle competitions download -c GiveMeSomeCredit`) and place it in `datasets/cs-training.csv` (or the project root).
3. Restart the kernel and run `credit_risk.ipynb` sequentially from top to bottom. Checkpoint variables are automatically serialized to `pickles/` via `save_workspace_variables()`.
