# Panic Disorder Detection: Imbalanced Classification

End-to-end machine-learning pipeline that predicts a panic-disorder diagnosis from structured clinical and lifestyle data. Built as the final project for **COSC 5610** (M.S. Computer Science, Marquette University, May 2025).

The interesting problem is **class imbalance**: only **4.3%** of the 120,000 records are positive. A model that always predicts "no" would score about 96% accuracy and be useless. The work compares resampling strategies and models to find a sensible trade-off between catching real cases (recall) and raising false alarms (precision).

## Results

Held-out test set: 24,000 rows, stratified 80/20 split. Hyperparameters were tuned with stratified cross-validation (`GridSearchCV`, scored on ROC-AUC).

| Model | Balancing | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | SMOTE | **0.970** | **0.641** | 0.668 | **0.655** | **0.987** |
| Logistic Regression | SMOTE | 0.915 | 0.336 | **0.998** | 0.502 | 0.965 |
| k-Nearest Neighbors | SMOTE | 0.897 | 0.286 | 0.943 | 0.439 | 0.967 |

Best cross-validated AUC: RF 0.985 · LR 0.965 · KNN 0.963.

**Reading the trade-off**
- **Random Forest + SMOTE** is the best all-round model: it misses 343 of 1,028 positives but raises only 379 false alarms.
- **Logistic Regression + SMOTE** catches almost every case (2 missed) at the cost of 2,030 false alarms. That is the right choice if the model is a *screening* step where a missed case is far costlier than a follow-up check.

Balancing strategies compared (test set):

| Model | Recall | F1 | AUC |
|---|---|---|---|
| RF + SMOTE | 0.668 | 0.655 | 0.987 |
| RF + random oversampling | 0.694 | 0.665 | 0.987 |
| RF + class weights | 0.518 | 0.592 | 0.987 |
| KNN, no balancing | 0.354 | 0.495 | 0.971 |
| KNN + SMOTE | 0.943 | 0.439 | 0.967 |
| LR + SMOTE / class weights / oversampling | ≈1.000 | 0.49–0.50 | 0.965 |

## Approach
1. **Exploration**: distributions of categorical features, a missing-value audit (Medical History, Psychiatric History and Substance Use are 25–33% empty), and class balance.
2. **Cleaning**: normalized categorical text, mode/median imputation, flagged out-of-range ages, checked for duplicates.
3. **Feature engineering**: an `Age_Bin` feature (teen / young / mid / senior) and a correlation check for redundant numeric features (none above 0.8).
4. **Pipelines**: `ColumnTransformer` (one-hot + scaling) inside an `imblearn` pipeline, so resampling happens **only within training folds** and test data never leaks into SMOTE.
5. **Models**: Logistic Regression, Random Forest, and KNN, each tuned with stratified CV.
6. **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and Random Forest feature importances.
7. **Ethics**: the report discusses how demographic features can encode existing disparities, so a screening model can amplify bias (for example, over-flagging one gender).

## Repository layout
```
data/panic_disorder_detection.csv   # 120,000 rows × 17 columns
notebooks/final_project.ipynb       # full pipeline with outputs
reports/final_report.pdf            # 7-page written report
requirements.txt
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/final_project.ipynb
```
The notebook reads the data from `../data/panic_disorder_detection.csv`.

## Known limitations
- The `Severity` and `Impact on Life` columns end up with no usable values after preprocessing, so the imputer drops them. They likely needed an ordinal mapping (mild → 1, moderate → 2, severe → 3) instead of numeric coercion. This is a worthwhile fix for a future version.
- Precision for the positive class stays modest across models. Threshold tuning on the Random Forest's probabilities is the obvious next step.
