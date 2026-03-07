# Dynamic Human Fatigue Prediction for Smart Production Scheduling

## 📌 Quick Overview

This project investigates whether wrist-worn motion data can be used to predict near-term assembly error risk, and whether dynamically timed micro-breaks can reduce errors without increasing total rest time. An assembly-like experimental setup is designed to simulate real production conditions, combining continuous IMU-based motion capture, feature extraction, predictive modeling, and initial proof-of-concept results.
> **Wrist-worn Sensor(IMU)** $\rightarrow$ **Multimodal Hybrid Deep Learning** $\rightarrow$ **Fatigue Monitoring** $\rightarrow$ **Assembly Error Prediction** $\rightarrow$ **Adaptive Intervention** *(AI-guided rest scheduling in iso-temporal conditions)*

---

## 1. Research Objectives

* **Predictive Modeling**: To assess whether wrist motion data can reliably predict assembly error risk within a 5-minute horizon.
* **Intervention Strategy Validation**: To evaluate the effectiveness of an AI-guided, fatigue-adaptive rest strategy compared with traditional fixed-interval rest under equal total rest time.
* **Architecture & Interpretability Benchmark**: To compare the robustness of tree-based and hybrid deep learning models, while ensuring prediction transparency through Explainable AI.

---

## 2. Experimental Environment and Apparatus

### 2.1 Hardware Configuration
| Category | Device/Component | Description |
| :--- | :--- | :--- |
| **Material System** | Source Containers | Two iron bowls containing "washers" and "beans" respectively. |
| | Target Containers | Two white bowls simulating the finished assembly area. |
| **Visual Induction** | PC Terminal | Running the experimental program, randomly switching digits (0-9) every 3 seconds. |
| **Sensor Node** | 9-axis IMU | Wrist-worn sensor collecting 3-axis Acceleration (Acc), Gyroscope (Gyro), and Euler Angles (Angle). |

<table border="0">
  <tr>
    <td align="center">
      <img src="assets/setup.png" width="300" alt="Experimental Setup">
      <br>
      <b>Figure 1: Experimental Setup</b>
    </td>
    <td align="center">
      <img src="assets/wrist_sensor.jpg" width="300" alt="Sensor Style">
      <br>
      <b>Figure 2: 9-axis Motion Sensor</b>
    </td>
  </tr>
</table>

### 2.2 Logic Design
* **Pace**: The screen will randomly generate a number between 0 and 9 every 3 seconds. (high-intensity repetitive task).
* **Task Branches**:
    * **Standard Task (Digit ≠ 3)**: Move "beans" directly into the white bowl.
    * **Anomaly Task (Digit = 3)**: Move "washer" into the white bowl first, then move "beans" into the white bowl.
![gif](assets/demo.gif)

---

## 3. Experimental Scenes Explanation 

### Theoretical Background
When performing highly repetitive tasks, the brain easily enters **"Autopilot"** mode. Over time, physiological and psychological fatigue accumulates, leading to a decline in **Vigilance**. At this point, if a "non-standard action" requiring extra attention appears, workers often make omissions or sequence errors due to inertia.

### Industrial Example 1: Car Part Assembly Line
* **Scenario Description**: A worker installs a **rubber seal** (protective strip) on a **car hood** (engine cover) every 15 seconds. This is a highly repetitive task that relies on muscle memory.
* **Special Task (The Change)**: Occasionally, a "Luxury Model" car arrives. For this specific car, the worker must apply **special glue** (adhesive) before installing the seal.
* **Error Risk**: After 2 hours of continuous work, the worker becomes very tired. Due to deep fatigue, their brain might "see" the luxury car but fail to react. They might follow their muscle memory and install the seal **without** the glue, skipping the extra step.

### Industrial Example 2: Electronics Assembly Line

* **Scenario Description:** A worker installs a heat sink (cooling metal piece) on standard circuit boards over and over. This is a fast, highly repetitive hand movement.
* **Special Task (The Change):** Occasionally, a "High-Performance" board arrives on the line. For this special board, the worker must manually apply a drop of thermal paste (cooling gel)  **before** installing the heat sink.
* **Error Risk:** After hours of doing the exact same standard motion, the worker experiences cognitive fatigue. They might just snap the heat sink on directly, skipping the crucial thermal paste step (an omission error).

---

## 4. Data Processing and Expected Results

### 4.1 Data Processing and Feature Extraction
Sensor data undergoes 15Hz low-pass filtering, outlier removal, and normalization. A total of **18 features** are extracted, with assembly errors (labeled via video review) serving as ground truth:
* **Time-Domain**: Standard Deviation (Std) of each axis and magnitude.
* **Kinematic**: Jerk Root Mean Square (Jerk_RMS), Peak Count.
* **Non-linear**: DFA-alpha ($DFA_\alpha$), etc.

### 4.2 Expected Outcomes
1.  **Algorithmic Evaluation Framework**: Validating the effectiveness of composite motion features in predicting assembly errors under high cognitive load; building and comparing static vs. temporal fatigue prediction models.
2.  **Adaptive Intervention Paradigm**: Proposing an adaptive rest algorithm that optimizes intervention timing without increasing total downtime, filling the gap between "fatigue monitoring" and "active intervention" in industry.

## 5. Proof of Concept (PoC)

Data collection for the first **3 participants** has been completed. After feature engineering, the processed data was used to train an **XGBoost** model. Initial results successfully validate the core hypothesis: **"Motion features (IMU data) can effectively predict assembly error risks."**

### Preliminary Model Performance Metrics:

<table border="0">
  <tr>
    <td align="center" width="50%">
      <img src="assets/Classification_Report.png" width="400" alt="Classification Report">
      <br>
      <b>Figure 3: Classification Report</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/ROC_Curve.png" width="400" alt="ROC Curve">
      <br>
      <b>Figure 4: ROC Curve</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/PR_Curve.png" width="400" alt="PR-AUC Curve">
      <br>
      <b>Figure 5: PR-AUC Curve</b>
    </td>
    <td align="center" width="50%">
      <img src="assets/Feature_Importance_Analysis.png" width="400" alt="Feature Importance">
      <br>
      <b>Figure 6: Feature Importance Analysis</b>
    </td>
  </tr>
</table>

### 6. Future Roadmap

* **Better Generalization**: Expand the dataset and use **Leave-One-Subject-Out (LOSO)** cross-validation to ensure the model works accurately for new users.

* **Model Optimization**: Improve feature engineering and tuning to compare **Tree-based models** (XGBoost) with **Deep Learning** (CNN-LSTM) for better accuracy.
* **System Integration**: Move from offline analysis to **real-time detection**, using fatigue data as dynamic **factors** for **AI scheduling** systems.

---

### 7. Vision

* **Integration into Real Production Systems**
* **Link to Scheduling Algorithms**
* **Larger Scale Validation**

---

### 8. Literature Review & Evidence Base

Extensive recent literature validates that wrist sensors (IMU) can successfully predict error risks:

* **Kazuo Yano** (**2009**, **Hitachi Research**) proved that the Zero-Crossing Rate (ZCR) of wrist accelerometers effectively measures human energy levels.
* **Zhang et al.** (**2021**, **Automation in Construction**) achieved 94% accuracy in detecting worker fatigue based on IMU jerk features.
* **Bangaru et al.** (**2022**, **Sensors**) demonstrated that wearable IMUs drastically outperform cardiovascular metrics in predicting physical fatigue, achieving over 92% accuracy compared to just 51% using heart rate. 
* **Khan et al.** (**2025**, **ITcon**) achieved over 90% accuracy in fatigue classification using wearable sensors across diverse physical conditions.
* **Albarrán Morillo et al.** (**2026**, **Safety Science**) achieved a high F1 score of 0.8793 in predicting worker fatigue using kinematic indicators.

A recent systematic review (**Naranjo et al.**, **2025**, **Sensors**) highlights the transformative potential of these technologies, noting that implementing wearable sensors can reduce workplace injuries by **25-30%**, increase productivity by **15-20%**, and decrease error rates in precision tasks by **35-40%**. However, the study also emphasizes that most current systems remain restricted to **passive data collection and early warning**, identifying a critical need for **AI-driven proactive interventions** to bridge the existing productivity gap.

---

## 📂 Reproducibility & Resources

To support the verification of the preliminary findings, the core implementation and a representative data sample have been uploaded to this repository:

* **Dataset Sample**: [processed_features_sample.csv](./processed_features_sample.csv) — A 1-hour segment of feature-engineered IMU data.
* **Source Code**:
    * [`src/03_preprocessing.py`](./src/03_preprocessing.py): Low-pass filtering (Butterworth),Valid task segmentation,Z-score normalization.
    * [`src/05_feature_engineering.py`](./src/05_feature_engineering.py): Calculation of acceleration and gyroscope magnitudes,Sliding window feature extraction (Statistical, Jerk, Peaks, DFA-alpha),Target label generation based on future error occurrences.
    * [`src/06_model_training.py`](./src/06_model_training.py): XGBoost training pipeline and performance evaluation.
