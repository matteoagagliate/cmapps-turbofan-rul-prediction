# 🔧 Predictive Models on NASA Turbofan Engine Degradation (C-MAPSS, FD004)

This project benchmarks a wide set of machine learning and deep learning models for **Remaining Useful Life (RUL)** prediction using NASA’s **C-MAPSS dataset**. We focus on **FD004**, the most challenging subset with **six operating conditions** and **two simultaneous fault modes (HPC + Fan degradation)**. This scenario stresses both robustness and predictive accuracy.

---

## 🛠 Project Goals
We designed a **reproducible, modular pipeline** that enables fair comparison of heterogeneous models under consistent preprocessing, training, and evaluation conditions.  
- **Linear regression** was used as a baseline.  
- Complexity was progressively increased (tree-based, feed-forward NN, recurrent NN, CNN) to test whether advanced architectures provide significant gains on FD004.  

---

## 📂 Pipeline Overview

1. **Data processing & filtering**  
   - Noise reduction via **Moving Average**, **Exponential Moving Average (EMA)**, **Savitzky–Golay filter**.  
   - Preserves local structure while denoising sensor streams.  

2. **Feature engineering**  
   - Scaling: **global vs condition-specific** (the latter proved crucial for recurrent models).  
   - **RUL clipping** to reduce label skew.  
   - **Lagged features** to encode short-term dependencies for non-sequential models.  
   - **Polynomial expansion** to capture nonlinear interactions.  

3. **Model training**  
   - Unified modular code (`DP`, `FE`, `MD`, `TN`, `UT`).  
   - Random seed management for reproducibility.  

4. **Evaluation metrics**  
   - **RMSE**, **R²**, **Mean Error (ME)**.  
   - A custom **Asymmetric MSE** (λ=2.0) penalizes **overestimation** more than underestimation — reflecting the safety-critical nature of prognostics.  

---

## 🤖 Implemented Models

- **Baselines**: Linear Regression, Polynomial Regression.  
- **Ensemble trees**: Random Forest, XGBoost.  
- **Neural networks**: Multilayer Perceptron (MLP).  
- **Recurrent models**: LSTM, GRU. 
- **Convolutional model**: 1D CNN.

---

## 🔍 Hyperparameter Tuning
- **Random Search** for each family, balancing exploration vs computation.  
- Each model family had its own dedicated search space, and the best configurations were stored to allow reproducibility.

---

## 📊 Results on FD004

Performance summary (Train / **Test** / CV).  
Metrics: **RMSE ↓**, **R² ↑**, **ME** (positive = overestimation, negative = underestimation).  

| Model | Split | RMSE | R² | ME |
|---|---|---:|---:|---:|
| **Linear** | Train | 20.98 | 0.73 | -0.00 |
|  | **Test** | **34.15** | **0.61** | **-4.28** |
|  | CV | 21.70 | 0.72 | 0.97 |
| **Polynomial** | Train | 17.90 | 0.81 | -0.00 |
|  | **Test** | **30.96** | **0.68** | **-7.04** |
|  | CV | 18.82 | 0.79 | 0.41 |
| **Random Forest** | Train | 16.26 | 0.84 | 0.02 |
|  | **Test** | **30.82** | **0.68** | **-5.84** |
|  | CV | 18.52 | 0.79 | 0.77 |
| **XGBoost** | Train | 16.03 | 0.84 | -0.00 |
|  | **Test** | **30.39** | **0.69** | **-7.37** |
|  | CV | 18.22 | 0.80 | 0.52 |
| **MLP** | Train | 17.05 | 0.82 | -0.42 |
|  | **Test** | **30.23** | **0.69** | **-7.63** |
|  | CV | 17.80 | 0.81 | 0.02 |
| **GRU** | Train | 15.18 | 0.87 | -4.74 |
|  | **Test** | **30.78** | **0.68** | **-13.71** |
|  | CV | 16.05 | 0.85 | -4.28 |
| **LSTM** | Train | 14.31 | 0.88 | 1.64 |
|  | **Test** | **25.95** | **0.77** | **-5.96** |
|  | CV | 14.86 | 0.87 | 1.71 |
| **CNN-1D** | Train | **12.81** | **0.91** | 0.97 |
|  | **Test** | **👉 25.78 (Best)** | **0.78** | **-7.11** |
|  | CV | **13.25** | **0.90** | 1.12 |

**Key insight:**  
- **CNN-1D achieved the best overall results**, slightly outperforming LSTM and significantly improving over baselines (−24.5% RMSE vs linear regression).  
- CNN combined strong **local feature extraction** (kernels capture short-term degradation signatures) with efficient training, giving the best generalization.  

---

## 🧠 Analysis & Model Insights

- **CNN-1D (Winner):**  
  - Excels by capturing **local temporal patterns** and composing them hierarchically via pooling.  
  - Approximates cumulative degradation trends efficiently.  
  - Offers best **bias–variance tradeoff**, and faster training than recurrent models.  

- **LSTM:**  
  - Very competitive, with excellent ability to model **long-term dependencies**.  
  - Performance strongly tied to **sequence length** and **scaling strategy** (condition-specific scaling crucial).  

- **GRU:**  
  - More efficient than LSTM but slightly worse generalization on FD004.  
  - Underperforms when operating conditions are heterogeneous.  

- **MLP:**  
  - Improves over trees, but lacks explicit temporal modeling.  
  - Tends to **underestimate RUL** consistently (ME < 0).  

- **XGBoost > Random Forest:**  
  - Better handling of high-variance regions near failure thresholds.  
  - Still suffers from train–test gap (overfitting to nonlinearities).  

- **Linear & Polynomial Regression:**  
  - Simple, interpretable, fast.  
  - Polynomial features reduce bias but risk overfitting, still unable to capture sequential degradation dynamics.  

**Error direction (ME):**  
- Most models **underestimate RUL** (negative ME), aligning with safety requirements.  
- We observed that employing an asymmetric loss further reinforces this tendency, penalizing overestimation more strongly and leading the models toward safer predictions.

---

## 👥 Authors
- Matteo Agagliate – [matteo.agagliate@studenti.polito.it](mailto:matteo.agagliate@studenti.polito.it)  
- Mattia Mosso – [mmosso3@gatech.edu](mailto:mmosso3@gatech.edu)  
