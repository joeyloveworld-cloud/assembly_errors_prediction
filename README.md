# Research on Assembly Error Prediction and Dynamic Rest Strategy Based on Motion Sensors

## Abstract
This research utilizes wearable sensor technology to monitor worker fatigue, explores the predictive capability of motion data for assembly risks, and validates the effectiveness of optimized rest strategies in reducing error rates.

---

## 1. Research Objectives
* **Predictive Modeling**: To verify whether wrist motion data (Acceleration, Angular Velocity, Orientation) can effectively predict assembly error risks within the next 5 minutes.
* **Intervention Strategy Validation**: To compare the "Optimized Rest Strategy" with the "Traditional Fixed-Interval Rest," evaluating the former's advantages in alleviating fatigue and reducing assembly failures.

---

## 2. Experimental Environment and Apparatus

### 2.1 Hardware Configuration
| Category | Device/Component | Description |
| :--- | :--- | :--- |
| **Material System** | Source Containers | Two iron bowls containing "washers" and "beans" respectively. |
| | Target Containers | Two ceramic bowls simulating the finished assembly area. |
| **Visual Induction** | PC Terminal | Running the experimental program, randomly switching digits (0-9) every 3 seconds. |
| **Sensor Node** | 9-axis IMU | Wrist-worn sensor collecting 3-axis Acceleration (Acc), Gyroscope (Gyro), and Euler Angles (Angle). |

### 2.2 Logic Design
* **Pace**: 3 seconds per cycle (high-intensity repetitive task).
* **Task Branches**:
    * **Standard Task (Digit ≠ 3)**: Move "beans" directly into the ceramic bowl.
    * **Anomaly Task (Digit = 3)**: Move "washer" into the ceramic bowl first, then move "beans" into the ceramic bowl.

---

## 3. Experimental Rules and Workflow
The experiment simulates a typical industrial assembly line. Participants must execute action sequences based on screen instructions:

* **When digit is "3" (Simulated Special Condition)**:
    * Path: Iron Bowl (Washer) → Ceramic Bowl A; Iron Bowl (Bean) → Ceramic Bowl B.
* **When digit is NOT "3" (Simulated Standard Condition)**:
    * Path: Iron Bowl (Bean) → Ceramic Bowl B.

---

## 4. Experimental Scenario Explanation (Case Study)
This experiment simulates a typical industrial scenario characterized by **"High Repetition, Low Mental Workload, but High Cognitive Demand."**

### Theoretical Background
When performing highly repetitive tasks, the brain easily enters **"Autopilot"** mode. Over time, physiological and psychological fatigue accumulates, leading to a decline in **Vigilance**. At this point, if a "non-standard action" requiring extra attention appears, workers often make omissions or sequence errors due to inertia.

### Industrial Comparison: Automotive Hood Seal Installation Line
* **Scenario Description**: A worker installs a standard seal every 15 seconds (corresponding to the "Digit ≠ 3" task). Due to the mechanical nature, it relies purely on muscle memory.
* **Anomaly Trigger**: Occasionally, a "Luxury Version" chassis appears, requiring a special adhesive to be applied before the seal (corresponding to the "Digit = 3" task).
* **Error Risk**: After working for 2 hours without rest, the worker may "see but not perceive" the luxury chassis due to deep fatigue, directly applying the seal by muscle memory and omitting the adhesive step.

---

## 5. Data Processing and Expected Results

### 5.1 Data Processing and Feature Extraction
Sensor data undergoes 15Hz low-pass filtering, outlier removal, and normalization. A total of **18 features** are extracted, with assembly errors (labeled via video review) serving as ground truth:
* **Time-Domain**: Standard Deviation (Std) of each axis and magnitude.
* **Kinematic**: Jerk Root Mean Square (Jerk_RMS), Peak Count.
* **Non-linear**: DFA-alpha ($DFA_\alpha$), etc.

### 5.2 Expected Outcomes
1.  **Algorithmic Evaluation Framework**: Validating the effectiveness of composite motion features in predicting omission errors under high cognitive load; building and comparing static vs. temporal fatigue prediction models.
2.  **Adaptive Intervention Paradigm**: Proposing an adaptive rest algorithm that optimizes intervention timing without increasing total downtime, filling the gap between "fatigue monitoring" and "active intervention" in industry.
