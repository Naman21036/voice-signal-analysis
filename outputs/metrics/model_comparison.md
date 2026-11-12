# Machine Learning Model Evaluation & Comparison

Course: EC24313 Artificial Intelligence and Machine Learning  
Assignment 2: Voice-Based Gender Classification  
Evaluation Protocol: Speaker-Independent Hold-Out Test Set (Zero Speaker Overlap)

## Model Comparison Table

| Classifier | Accuracy | Sensitivity (Recall) | Precision | Specificity | F1 Score | ROC AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
| **Support Vector Machine** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | TP=16, TN=5, FP=0, FN=0 |
| **Random Forest** | 80.95% | 100.00% | 80.00% | 20.00% | 0.8889 | 1.0000 | TP=16, TN=1, FP=4, FN=0 |
| **Neural Network (MLP)** | 95.24% | 100.00% | 94.12% | 80.00% | 0.9697 | 0.9875 | TP=16, TN=4, FP=1, FN=0 |

### Evaluation Convention
- **Class 1 (Positive):** Boys (label 0)
- **Class 2 (Negative):** Girls (label 1)
- **Sensitivity:** Fraction of true boy recordings correctly detected ($TP / (TP + FN)$).
- **Specificity:** Fraction of true girl recordings correctly detected ($TN / (TN + FP)$).
- **Precision:** Fraction of boy predictions that are actually boys ($TP / (TP + FP)$).
- **F1 Score:** Harmonic mean of precision and sensitivity.
- **ROC AUC:** Area under Receiver Operating Characteristic curve.
