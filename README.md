# Quantum vs Classical Machine Learning for Credit Card Fraud Detection

This project compares three classical machine learning models (Random Forest,
XGBoost, SVM) against two quantum machine learning models (QSVM, QNN) on the
task of detecting fraudulent credit card transactions.

## Dataset

The dataset is the standard Kaggle "Credit Card Fraud Detection" dataset:
284,807 transactions, 492 of which are fraud (0.17% fraud rate). Features are
PCA-anonymized (V1-V28) plus Time and Amount. The raw CSV is not included in
this repo due to its size; it can be downloaded from Kaggle.

## Approach

Classical models (Random Forest, XGBoost, SVM) are trained and evaluated on
the full dataset, using SMOTE for class imbalance (applied only to the
training split, after the train/test split, to avoid data leakage) and
threshold tuning to maximize fraud-class F1.

Quantum models (QSVM, QNN) are trained on smaller subsets, since quantum
circuit simulation does not scale to hundreds of thousands of rows the way
classical algorithms do. QSVM uses 10 features with a ZZFeatureMap kernel;
QNN is a hybrid classical-quantum network using a 3-qubit PauliFeatureMap and
RealAmplitudes ansatz. Their results should be read as a proof of concept
rather than a like-for-like comparison at full scale. Both were trained once
on a classical simulator (training took several days), and their results are
reused here rather than recomputed.

## Results

| Model         | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | AUC    |
|---------------|----------|------------------|---------------|----------|--------|
| Random Forest | 99.95%   | 90.6%            | 78.6%         | 84.2%    | 0.970  |
| XGBoost       | 99.96%   | 96.4%            | 81.6%         | 88.4%    | 0.981  |
| SVM           | 99.64%   | 30.7%            | 88.8%         | 45.7%    | 0.955  |
| QSVM          | 95.0%    | 83.0%            | 87.0%         | 85.0%    | N/A*   |
| QNN           | 76.7%    | 79.3%            | 72.2%         | 75.6%    | 0.815  |

QSVM was trained without probability estimates, so AUC is not available.

Classical models are evaluated on the real imbalanced test set (56,962
transactions, 98 fraud). QSVM is evaluated on a smaller undersampled test set
(591 transactions, 98 fraud), and QNN on an even smaller balanced test set
(180 transactions), because of simulator limits.

## Notebooks

- `fraud_detection_pipeline.ipynb` — the final, clean pipeline: EDA, all five
  models, evaluation, and the results comparison. This is the one to read.
- `original_exploration.ipynb` — the original development notebook, kept
  unedited for transparency. It includes early iterations and mistakes that
  were caught and corrected in the final version, including a hardcoded
  threshold value in an early XGBoost run and a data leakage bug from
  applying SMOTE before the train/test split.

## Repository structure
