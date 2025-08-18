# 🔧 Predictive Models on NASA Turbofan Engine Degradation Dataset  

This project evaluates a wide range of machine learning and deep learning models for the prediction of the **Remaining Useful Life (RUL)** of turbofan engines. The study is based on the **NASA C-MAPSS dataset**, a well-established benchmark in the prognostics community.  

---

## 🛠 Project Overview  
Our objective was to design, implement, and compare different predictive approaches within a unified, reproducible pipeline. As a reference point, we adopted **linear regression** as the baseline model, then extended the analysis to more sophisticated methods in order to assess how different architectures perform under the same experimental conditions.  

The experiments focus on **FD004**, the most challenging C-MAPSS subset: it contains multiple operational conditions and two simultaneous fault modes (HPC degradation and fan degradation). This complexity makes FD004 an ideal testbed for evaluating both the robustness and accuracy of RUL prediction models.  

---

## 📂 Methodology & Pipeline  
The system is implemented as a modular Python framework. The pipeline covers the entire process from raw sensor readings to final evaluation:  

1. **Data processing & filtering** – noise reduction through Moving Average, Exponential Moving Average, or Savitzky–Golay filters.  
2. **Feature engineering** – scaling (global or condition-specific), clipping of RUL values, lagged variables to encode temporal dependencies, and polynomial expansion to capture nonlinear interactions.  
3. **Model training & evaluation** – each model is trained with consistent preprocessing and compared on common metrics.  
4. **Evaluation metrics** – RMSE and R² are reported, together with a custom **Asymmetric MSE** that penalizes RUL overestimation more heavily to reflect the safety-critical nature of the task.  

This design ensures that experiments are **reproducible and comparable** across all models.  

---

## 🤖 Implemented Models  
Alongside the linear regression baseline, we explored a diverse set of algorithms, chosen to represent different levels of complexity and modeling power:  

- **Random Forest** and **XGBoost** to capture nonlinearities and interactions between sensors.  
- **Multilayer Perceptron (MLP)** as a first step into neural models capable of modeling nonlinear mappings.  
- **Recurrent architectures (LSTM, GRU)** to explicitly learn temporal dependencies in the sensor time series.  
- **1D Convolutional Neural Network (CNN)** to extract local temporal patterns and correlations among sensors with efficient training.  

The choice of models reflects the goal of **progressively testing whether more advanced architectures significantly outperform simple baselines** on FD004.  

---

## 🔍 Hyperparameter Tuning  
Hyperparameters were optimized using **Random Search**, which offered a practical trade-off between exploration of the search space and computational cost. Separate searches were conducted for each model family, allowing fair comparisons of their best-performing configurations.  

---

## 📊 Results & Insights  
The experiments confirmed that the **linear regression baseline** provides fast and interpretable results, but struggles with the complex degradation dynamics of FD004.  
- **Random Forest** and **XGBoost** offered clear improvements, especially in capturing nonlinear behaviors, though they plateaued in performance.  
- **MLP** achieved stronger predictive accuracy than tree-based models but required careful regularization.  
- **LSTM** and **GRU** delivered the **best results overall**, demonstrating their strength in modeling long-term temporal dependencies in noisy sensor data.  
- **1D CNN** performed competitively, approaching recurrent models while offering faster training times, though it was slightly less accurate in the long-term RUL predictions.  

In summary, **recurrent neural networks outperformed all other methods**, validating their suitability for complex prognostics tasks. At the same time, tree-based models and CNNs provided valuable trade-offs in terms of interpretability and efficiency.  

---

## 👥 Authors  

- Matteo Agagliate – [matteo.agagliate@studenti.polito.it](mailto:matteo.agagliate@studenti.polito.it) 
- Mattia Mosso – [mmosso3@gatech.edu](mailto:mmosso3@gatech.edu)  



