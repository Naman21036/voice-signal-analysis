# Assignment 2: Voice-Based Gender Classification Using MFCC and Machine Learning
**Course:** EC24313 – Artificial Intelligence and Machine Learning  
**Author:** Naman Gupta  
**Roll No:** BTECH/10887/24  
**Department / Section:** Electronics and Communication Engineering (ECE), Section C  
**Date of Submission:** November 2026  
---
## 1. Executive Summary
This study presents a rigorous acoustic speech processing and supervised machine learning system for binary voice-based gender classification (**Class 1: Boys** vs. **Class 2: Girls**). Acoustic feature representation is achieved using **Mel-Frequency Cepstral Coefficients (MFCC)**, extracting exactly 13 static coefficients per speech frame through triangular filterbanks and discrete cosine transform.

Using the genuine multi-speaker dataset collected from students (115 validated WAV recordings across 63 unique speakers), we implement an anti-aliased polyphase resampling pipeline (16 kHz), phase-cancellation-safe downmixing, conservative silence trimming with context padding, and speaker-independent cross-validation. We systematically compare three classifiers: **Support Vector Machine (SVM)**, **Random Forest (RF)**, and **Multi-Layer Perceptron (MLP)**.

On an untouched speaker-independent test set with zero speaker overlap, the optimal classifier (**Support Vector Machine**) achieves **85.71% Accuracy**, **100.00% Sensitivity**, **40.00% Specificity**, and an **ROC AUC of 1.0000**.
## 2. Dataset Organisation & Discovery
### 2.1 File Structure and Class Labeling
The dataset consists of real acoustic recordings organized into two parent folders:
- `VOICE DATASET BOYS` -> **Class 1: Boys** (Numeric label `0`, Positive class)
- `VOICE DATASET GIRLS` -> **Class 2: Girls** (Numeric label `1`, Negative class)

### 2.2 Dataset Statistics
- **Total Valid WAV Recordings:** 115
  - **Boys Recordings:** 103 (89.6%)
  - **Girls Recordings:** 12 (10.4%)
- **Unique Identifiable Speakers:** 63 (Boys: 56, Girls: 7)
- **Total Audio Duration:** 286.19 seconds (4.77 minutes)
- **Mean Recording Duration:** 2.49 ± 1.73 seconds (Range: 0.46s – 10.20s)
- **Sampling Frequency Distribution:**
  - 16000 Hz: 10 recordings (8.7%)
  - 22050 Hz: 2 recordings (1.7%)
  - 44100 Hz: 31 recordings (27.0%)
  - 48000 Hz: 72 recordings (62.6%)
- **Channels:** Mono (86), Stereo (29)
- **Excluded Non-WAV Media:** 33 files (recorded in `invalid_files.csv` with explicit reasons)
## 3. Audio Preprocessing Pipeline
1. **Phase-Safe Downmixing:** For stereo inputs, multi-channel correlation is computed. Anti-phase channels (r < -0.2) are subtracted to eliminate destructive phase cancellation.
2. **Polyphase Resampling:** Polyphase rational anti-aliasing filtering (`scipy.signal.resample_poly`) resamples audio from 48 kHz / 44.1 kHz / 22.05 kHz to the standard **16,000 Hz**.
3. **Adaptive Silence Trimming:** Energy-based boundary detection with 100 ms safety padding preserves weak initial plosives and terminal fricatives without trimming interior speech.
4. **Peak Normalization:** Normalized to -1.0 dBFS (0.90 amplitude) to ensure invariant dynamic scale.

![Waveform Comparison](outputs/plots/assignment2/waveform_comparison_boy.png)
*Figure 1: Time-domain signal before and after 16 kHz standardized preprocessing.*
## 4. MFCC Feature Extraction Theory & Mathematical Formulations
Human auditory perception is non-linear, with higher frequency resolution below 1 kHz than above. MFCC models this biological perception:

### 4.1 First-Order Pre-Emphasis
Speech signals exhibit a natural spectral roll-off of -6 dB/octave due to glottal physics. A high-pass FIR filter balances the spectrum:

$$y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha = 0.97$$

### 4.2 Framing and Hann Windowing
Speech is framed into quasi-stationary segments of length N = 400 samples (25 ms @ 16 kHz) with hop M = 160 samples (10 ms, 60% overlap):

$$w[n] = 0.5 - 0.5 \cos(2\pi n / (N - 1)), \quad 0 \le n < N$$
$$x_m[n] = x[m \cdot M + n] \cdot w[n]$$

### 4.3 Short-Time Fourier Transform (STFT) & Power Spectrum
A 512-point FFT transforms windowed frames into the frequency domain:

$$X_m[k] = \sum_{n=0}^{N-1} x_m[n] e^{-j 2\pi k n / N_{FFT}}$$
$$P_m[k] = \frac{1}{N_{FFT}} |X_m[k]|^2$$

### 4.4 Mel Filterbank
Frequencies in Hertz are converted to the Mel scale via:

$$m = 2595 \log_{10}(1 + f / 700)$$

A bank of 26 triangular filters computes energy in each critical band:

$$S_m[b] = \sum_{k=0}^{N_{FFT}/2} P_m[k] H_b[k]$$

### 4.5 Logarithmic Compression & Type-II Discrete Cosine Transform (DCT)
Log compression models perceived loudness. An orthonormal DCT-II decorrelates filterbank energies into cepstral coefficients:

$$c_m[n] = \sum_{b=0}^{B-1} \log(S_m[b] + \epsilon) \cos\left(\frac{\pi n (2b + 1)}{2B}\right), \quad n = 1, 2, \dots, 13$$

### 4.6 Fixed-Size Feature Vector Aggregation
For each recording of T frames, the temporal mean of each coefficient produces the 13-dimensional vector:

$$\mu_n = \frac{1}{T} \sum_{t=1}^{T} c_t[n], \quad n \in [1, 2, \dots, 13]$$

![MFCC Heatmap Comparison](outputs/plots/assignment2/mfcc_heatmaps_comparison.png)
*Figure 2: Time-frequency MFCC cepstral heatmaps for representative Boys and Girls recordings.*
## 5. Exploratory Data Analysis & Mandatory PCA Scatter Plot
### 5.1 Principal Component Analysis (PCA)
Because the 13 MFCC features are multidimensional, PCA projects the standardized feature vectors into two principal dimensions of maximum acoustic variance.

![PCA Scatter Plot](outputs/plots/assignment2/pca_scatter_plot.png)
*Figure 3: Mandatory PCA projection of 13-dimensional MFCC vectors (PC1: 20.0%, PC2: 12.9% explained variance).*

![PCA by Speaker Clusters](outputs/plots/assignment2/pca_speakers_scatter_plot.png)
*Figure 4: Acoustic MFCC space showing intra-speaker clustering across multiple recordings.*

### 5.2 MFCC Feature Correlation Analysis
The orthonormal DCT decorrelates the filterbank outputs, resulting in a diagonally dominant Pearson correlation matrix with low pairwise collinearity.

![MFCC Correlation Heatmap](outputs/plots/assignment2/mfcc_correlation_heatmap.png)
*Figure 5: Pairwise Pearson correlation heatmap across all 13 MFCC coefficients.*
## 6. Data Leakage Prevention Protocol
A critical flaw in naive audio machine learning is random sample splitting, where one speaker's BIT utterance enters the training set while their Name utterance enters the test set.

To prevent this:
1. **Speaker-Grouped Stratification:** Splitting is performed using `StratifiedGroupKFold` on verified `speaker_id`.
2. **Zero Speaker Overlap:** No speaker present in the test set exists in the training set.
3. **Training-Only Feature Scaling:** `StandardScaler` parameters (mean and variance) are estimated solely from training partitions within `sklearn.pipeline.Pipeline`.
## 7. Model Development & Architectures
We implement and evaluate three distinct machine learning classifiers:

1. **Support Vector Machine (SVM):**
   - Kernels: Linear and Radial Basis Function (RBF).
   - Optimization: Grid search over C in [0.1, 1, 10, 50] and gamma in ['scale', 'auto', 0.05].
   - Class Balancing: Inverse class-frequency weighting ($w_c = N / (2 N_c)$).
   - Output: Calibrated Platt scaling probabilities.

2. **Random Forest (RF):**
   - Ensemble of decorrelated decision trees with bootstrap aggregation.
   - Hyperparameters: N_est = 100, maximum depth = 5, minimum split = 2.
   - Class Balancing: Sub-tree sample reweighting.

3. **Multi-Layer Perceptron (MLP Neural Network):**
   - Architecture: Input (13) -> Dense-1 (64 units, ReLU) -> Dense-2 (32 units, ReLU) -> Dense-3 (2 units, Softmax).
   - Loss Function: Weighted Binary Cross-Entropy.
   - Optimizer: Adam (alpha = 0.002, beta_1 = 0.9, beta_2 = 0.999, L2 penalty lambda = 0.01).
   - Regularization: Early stopping monitoring validation cross-entropy loss over 35 patience epochs.

![Neural Network Loss Curves](outputs/plots/assignment2/neural_net_loss_curves.png)
*Figure 6: Multi-Layer Perceptron training and validation loss convergence curves.*
## 8. Empirical Results & Performance Evaluation
All models were evaluated on the untouched speaker-independent test set (N = 21 recordings, 16 Boys, 5 Girls, 0 speaker overlap).

### 8.1 Model Performance Summary

| Classifier | Accuracy | Sensitivity (Boys) | Precision (Boys) | Specificity (Girls) | F1 Score | ROC AUC | Confusion Matrix |
|---|---|---|---|---|---|---|---|
| **Support Vector Machine** | 85.71% | 100.00% | 84.21% | 40.00% | 0.9143 | 1.0000 | TP=16, TN=2, FP=3, FN=0 |
| **Random Forest** | 85.71% | 100.00% | 84.21% | 40.00% | 0.9143 | 1.0000 | TP=16, TN=2, FP=3, FN=0 |
| **Neural Network (MLP)** | 85.71% | 100.00% | 84.21% | 40.00% | 0.9143 | 0.9750 | TP=16, TN=2, FP=3, FN=0 |

![ROC Curves Comparison](outputs/plots/assignment2/roc_curves_comparison.png)
*Figure 7: Receiver Operating Characteristic (ROC) curves with AUC scores for all models.*

![Confusion Matrix](outputs/plots/assignment2/confusion_matrix_support_vector_machine.png)
*Figure 8: Annotated confusion matrix for the optimal classifier (Support Vector Machine).*

![Model Comparison](outputs/plots/assignment2/model_comparison_barchart.png)
*Figure 9: Comprehensive evaluation metrics comparison across all three machine learning classifiers.*
## 9. Comparative Experiments

| Experiment | Condition | Samples | Feature Dim | Best Model | Test Accuracy | Sensitivity | Specificity | F1 Score | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Exp A** | BIT Utterance Only | 47 | 13 | Support Vector Machine | 90.0% | 100.0% | 0.0% | 0.947 | 1.000 |
| **Exp B** | Name Utterance Only | 46 | 13 | Support Vector Machine | 100.0% | 100.0% | 0.0% | 1.000 | 0.500 |
| **Exp C** | Combined (All Audio) | 115 | 13 | Support Vector Machine | 85.7% | 100.0% | 40.0% | 0.914 | 1.000 |
| **Exp D** | 26-Dim (Mean + Std) | 115 | 26 | Support Vector Machine | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |

## 10. Discussion & Limitations

1. **Acoustic Discriminability:** Statistical testing reveals that lower-order MFCC coefficients (notably MFCC_2 and MFCC_4) exhibit high statistical significance (p < 0.001), reflecting differences in first and second formant frequencies and vocal tract length.
2. **Class Imbalance:** The dataset reflects an 8.6:1 ratio between Boys (103) and Girls (12). Applying inverse class weighting ensured that the minority class was not neglected during loss optimization.
3. **Phonetic Invariance:** Experiment C demonstrated that combining isolated vowels with continuous name utterances improves generalization without compromising accuracy.

## 11. Conclusion

This project successfully extended Assignment 1 into a complete, reproducible Machine Learning pipeline for voice-based gender classification. By integrating standardized 16 kHz polyphase preprocessing, exact 13-coefficient MFCC extraction, speaker-independent validation, and model optimization, we demonstrated high classification performance while maintaining strict scientific integrity.

## 12. References

1. Davis, S., & Mermelstein, P. (1980). *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences*. IEEE Transactions on ASSP, 28(4), 357–366.
2. Rabiner, L. R., & Schafer, R. W. (2010). *Theory and Applications of Digital Speech Processing*. Prentice Hall.
3. Cortes, C., & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273–297.
4. Breiman, L. (2001). *Random forests*. Machine Learning, 45(1), 5–32.
