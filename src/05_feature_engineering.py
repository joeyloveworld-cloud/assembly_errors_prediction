import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import timedelta, datetime
import time
import tkinter as tk
from tkinter import filedialog
import os
import sys

"""
Feature Engineering Script for Sensor Data
This script performs:
1. Data loading (Merged dataset with error labels)
2. Calculation of acceleration and gyroscope magnitudes
3. Sliding window feature extraction (Statistical, Jerk, Peaks, DFA-alpha)
4. Target label generation based on future error occurrences
"""

# ==========================================
# CONFIGURATION
# ==========================================
# Sliding Window Parameters
OBSERVATION_WINDOW = timedelta(minutes=5)      # X: Past observation window
PREDICTION_DELAY = timedelta(minutes=0)        # Blind zone before prediction
PREDICTION_WINDOW = timedelta(minutes=5)       # y: Future prediction window (span)
STEP_SIZE = timedelta(seconds=30)               # Sliding step size
SAMPLING_RATE = 50.0                           # Sampling rate in Hz

# Target columns mapping
# Note: Keep the dictionary values in Chinese if they match your actual dataset headers
COLS_TO_EXTRACT = {
    'Acc_X': '加速度X(g)', 'Acc_Y': '加速度Y(g)', 'Acc_Z': '加速度Z(g)', 'Acc_Mag': 'Acc_Mag',
    'Gyro_X': '角速度X(°/s)', 'Gyro_Y': '角速度Y(°/s)', 'Gyro_Z': '角速度Z(°/s)', 'Gyro_Mag': 'Gyro_Mag'
}
# ==========================================

# ---------------------------------------------------------
# 1. Manually select the merged dataset to process
# ---------------------------------------------------------
# Hide the main Tkinter window
root = tk.Tk()
root.withdraw()

print("Step 1: Please select the dataset for feature extraction (e.g., Final_LSTM_Dataset_xxx.csv)...")
file_path = filedialog.askopenfilename(
    title="Select Dataset with Merged Error Labels",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected, program terminated.")
    sys.exit()

print(f"Loading dataset: {file_path}")
# Read data (read-only operation, will not modify the original file)
try:
    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='gbk', low_memory=False)

df['System_Time'] = pd.to_datetime(df['System_Time'])

# ---------------------------------------------------------
# 2. Pre-computations & Algorithm Definitions
# ---------------------------------------------------------
print("Pre-computing Acceleration and Gyroscope Magnitudes...")
# Pre-calculate Vector Magnitude for Acceleration and Gyroscope
df['Acc_Mag'] = np.sqrt(df['加速度X(g)']**2 + df['加速度Y(g)']**2 + df['加速度Z(g)']**2)
df['Gyro_Mag'] = np.sqrt(df['角速度X(°/s)']**2 + df['角速度Y(°/s)']**2 + df['角速度Z(°/s)']**2)

# Pure Numpy implementation of DFA-alpha algorithm (no special environment required)
def calculate_dfa(signal):
    y = np.cumsum(signal - np.mean(signal))
    box_sizes = np.logspace(1, np.log10(len(y)//10), 20).astype(int)
    box_sizes = np.unique(box_sizes)
    
    fluctuations = []
    for scale in box_sizes:
        segments = len(y) // scale
        y_reshaped = y[:segments * scale].reshape((segments, scale))
        x = np.arange(scale)
        rms = 0
        for i in range(segments):
            p = np.polyfit(x, y_reshaped[i], 1)
            trend = np.polyval(p, x)
            rms += np.mean((y_reshaped[i] - trend)**2)
        fluctuations.append(np.sqrt(rms / segments))
    
    valid = np.array(fluctuations) > 0
    if np.sum(valid) > 2:
        coeffs = np.polyfit(np.log10(box_sizes[valid]), np.log10(np.array(fluctuations)[valid]), 1)
        return coeffs[0]
    return np.nan

# ---------------------------------------------------------
# 3. Core Sliding Window Feature Extraction
# ---------------------------------------------------------
# Define the scanning time range (strictly truncate ends to prevent out-of-bounds)
start_time = df['System_Time'].min() + OBSERVATION_WINDOW
end_time = df['System_Time'].max() - (PREDICTION_DELAY + PREDICTION_WINDOW)

print(f"Total data span: {df['System_Time'].min()} to {df['System_Time'].max()}")
print(f"Sliding feature extraction starts: Sliding from {start_time} to {end_time}")

current_time = start_time
features_list = []

while current_time <= end_time:
    # ---------------------------------------------------
    # Step 3.1: Segment data for the past observation window
    # ---------------------------------------------------
    past_mask = (df['System_Time'] > current_time - OBSERVATION_WINDOW) & (df['System_Time'] <= current_time)
    past_window = df[past_mask]
    
    # [Core Step: Physical filtering] Keep only data where Is_Valid_Task == 1 within this window
    valid_past = past_window[past_window['Is_Valid_Task'] == 1]
    
    # Fault tolerance mechanism: If this window happens to contain very few valid actions 
    # (e.g., less than 1 minute of data = 3000 rows at 50Hz), we discard this sample to prevent distortion
    if len(valid_past) < 3000:
        current_time += STEP_SIZE
        continue
        
    feat_dict = {}
    feat_dict['Time_Stamp'] = current_time # Record the timestamp for this sample
    
    # ---------------------------------------------------
    # Step 3.2: Calculate X features for the observation window
    # ---------------------------------------------------
    for name, col_name in COLS_TO_EXTRACT.items():
        data = valid_past[col_name].values
        
        # 1. Calculate Standard Deviation (Std)
        feat_dict[f'{name}_Std'] = np.std(data)
        
        # 2. Calculate Jerk RMS - multiplied by sampling rate to convert to physical rate of change
        jerk = np.diff(data) * SAMPLING_RATE
        feat_dict[f'{name}_Jerk_RMS'] = np.sqrt(np.mean(jerk**2))
        
    # 3. Calculate Peak Count
    acc_mag = valid_past['Acc_Mag'].values
    peaks, _ = find_peaks(acc_mag, distance=50, height=np.mean(acc_mag))
    feat_dict['Peak_Count'] = len(peaks)
    
    # 4. Calculate DFA Alpha index (measures rhythm fluctuations)
    feat_dict['DFA_alpha'] = calculate_dfa(acc_mag)
    
    # ---------------------------------------------------
    # Step 3.3: Generate future prediction label y
    # ---------------------------------------------------
    future_start = current_time + PREDICTION_DELAY
    future_end = future_start + PREDICTION_WINDOW
    future_mask = (df['System_Time'] > future_start) & (df['System_Time'] <= future_end)
    future_window = df[future_mask]
    
    # If Error_Mark occurs at least once in the future window, label is 1 (Error)
    if future_window['Error_Mark'].sum() > 0:
        feat_dict['y_Label'] = 1
    else:
        feat_dict['y_Label'] = 0
        
    # Append this sample to the list
    features_list.append(feat_dict)
    
    # Slide forward by the specified step size
    current_time += STEP_SIZE

# ---------------------------------------------------------
# 4. Save the final extracted feature dataset
# ---------------------------------------------------------
result_df = pd.DataFrame(features_list)

# Get current timestamp: Month, Day, Hour, Minute (e.g., 0228_1530)
timestamp = datetime.now().strftime("%m%d_%H%M")

# Dynamic directory path handling (Creates an 'extracted_features' folder next to the input file)
base_dir = os.path.dirname(file_path)
save_dir = os.path.join(base_dir, "extracted_features")

# Automatically create the directory if it doesn't exist
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
        print(f"Target folder created automatically: {save_dir}")
    except Exception as e:
        print(f"Failed to create folder, saving to current directory. Error: {e}")
        save_dir = "."

# Generate final filename with timestamp
save_filename = f"ML_Features_Dataset_{timestamp}.csv"
save_path = os.path.join(save_dir, save_filename)

# Export feature table (using utf-8-sig to prevent character encoding issues in Excel)
try:
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print("-" * 50)
    print(f"Feature engineering completed successfully! Generated {len(result_df)} ML samples.")
    print(f"Label distribution:\n{result_df['y_Label'].value_counts()}")
    print(f"\nFeature file successfully saved to: {save_path}")
except Exception as e:
    print(f"Error saving file: {e}")