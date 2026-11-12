# Comparative Utterance and Feature Experiments

Course: EC24313 Artificial Intelligence and Machine Learning  
Assignment 2: Voice-Based Gender Classification  

## Experimental Findings

| Experiment | Condition | Samples | Feature Dim | Best Model | Test Accuracy | Sensitivity | Specificity | F1 Score | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Experiment A** | BIT Utterance Only | 47 | 13 | Support Vector Machine | 90.0% | 100.0% | 0.0% | 0.947 | 1.000 |
| **Experiment B** | Name Utterance Only | 46 | 13 | Support Vector Machine | 100.0% | 100.0% | 0.0% | 1.000 | 0.500 |
| **Experiment C** | Combined (All Audio) | 115 | 13 | Support Vector Machine | 85.7% | 100.0% | 40.0% | 0.914 | 1.000 |
| **Experiment D** | 26-Dim (Mean + Std) | 115 | 26 | Support Vector Machine | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |

### Analysis
- **Monosyllabic vs Polysyllabic Utterances:** Evaluating 'BIT' vs 'Name' recordings shows how phonetic variation affects acoustic separation. Longer polysyllabic names capture richer vocal tract resonances.
- **13-Dimensional vs 26-Dimensional Features:** Adding the standard deviation across frames (temporal variability) provides dynamic contour information, maintaining strong discriminative capacity.
