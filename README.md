# Panic Disorder Detection: An Imbalanced Classification Pipeline

An end-to-end machine learning pipeline that predicts a binary **Panic Disorder Diagnosis** from a 120,000-row synthetic clinical/lifestyle dataset in which only **~4.3% of rows are positive**. The core problem is **class imbalance**: a model that always predicts "no" is 95.7% accurate and useless.

Graduate final project for COSC 5610 (Machine Learning), Marquette University, May 2025.

## What's here

| File | Description |
|---|---|
| `final_project.ipynb` | The full notebook: EDA, cleaning, feature engineering, modeling, and evaluation |
| `panic_disorder_detection.csv` | The dataset (120k rows, 17 columns) |
| `Abir Golam_final_report.pdf` | The written report: methodology, results, and ethical reflection |
| `final_project.pdf` | The project brief |

## Approach

1. **EDA:** distributions of categorical risk factors, plus a class-balance check (114,858 negative vs. 5,142 positive)
2. **Cleaning:** median and mode imputation, lowercasing and whitespace normalization of categorical values, de-duplication, and age outlier clipping
3. **Feature engineering:** an `Age_Bin` bucket and an `Impact × Support` interaction term, followed by one-hot encoding and z-scaling
4. **Leakage-safe pipelines:** preprocessing and resampling are wrapped in `imblearn` pipelines, so oversampling only ever touches training folds
5. **Imbalance strategies compared:** SMOTE, random oversampling, and class weighting
6. **Models:** Logistic Regression, Random Forest, and KNN, tuned with 5-fold stratified grid search on an 80/20 stratified split

## Results (held-out test set, n = 24,000)

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **RF + Random Oversampling** | 0.970 | 0.694 | **0.665** | **0.987** |
| RF + SMOTE | 0.970 | 0.668 | 0.655 | 0.987 |
| RF + class_weight | 0.969 | 0.518 | 0.592 | 0.987 |
| LR + SMOTE | 0.915 | **0.998** | 0.502 | 0.965 |
| KNN + SMOTE | 0.897 | 0.943 | 0.439 | 0.967 |
| KNN (no balancing) | 0.969 | 0.354 | 0.495 | 0.971 |

**Takeaway:** accuracy barely moves across models, so it is the wrong metric here. Logistic regression with resampling catches nearly every positive case (recall ≈ 1.0) at the cost of many false alarms. The random forest gives the best precision/recall balance (F1). Which one to choose is a product decision about the cost of a missed case versus a false alarm, not purely a modeling one.

## Running it

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
jupyter notebook final_project.ipynb
```

## Tech stack

Python · pandas · scikit-learn · imbalanced-learn · matplotlib · seaborn · Jupyter
