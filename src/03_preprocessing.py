import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import sys
from datetime import datetime
from scipy.signal import butter, filtfilt

"""
Data Preprocessing Script for Sensor Data
This script performs:
1. Data loading (CSV/Excel)
2. Low-pass filtering (Butterworth)
3. Valid task segmentation based on 'Sync_Mark'
4. Z-score normalization
"""

# ==========================================
# CONFIGURATION
# ==========================================
CUTOFF_FREQ = 15.0  # Cutoff frequency in Hz
FILTER_ORDER = 2    # Butterworth filter order
# Note: Modify these column names to match your actual dataset headers
TARGET_COLUMNS = [
    '加速度X(g)', '加速度Y(g)', '加速度Z(g)', 
    '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)', 
    '角度X(°)', '角度Y(°)', '角度Z(°)'
]
# ==========================================

# ---------------------------------------------------------
# 1. Manually select the file to process
# ---------------------------------------------------------
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

# Pop up file selection dialog
file_path = filedialog.askopenfilename(
    title="Please select the data file to process",
    filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected, program terminated.")
    sys.exit()

print(f"Reading file: {file_path}")

# Support reading CSV or Excel formats
if file_path.endswith('.csv'):
    # Add encoding and mixed type handling to prevent parsing errors for Chinese headers
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
else:
    df = pd.read_excel(file_path)

# Extract System_Time and convert to datetime object, parses perfectly even with milliseconds
df['System_Time'] = pd.to_datetime(df['System_Time'])

# ---------------------------------------------------------
# 2. Signal Filtering: Apply a 2nd-order Butterworth low-pass filter with a 15 Hz cutoff frequency
# ---------------------------------------------------------
print("Applying low-pass filter...")

# Specify sensor data columns to be filtered (automatically matched based on CSV headers)
# Note: Kept in Chinese to match the actual headers in the input file
target_cols = TARGET_COLUMNS
# Keep only the columns that actually exist in the file to avoid errors
filter_cols = [col for col in target_cols if col in df.columns]

# Dynamically estimate sampling frequency (fs): estimated by dividing total time by the number of data rows
total_time_seconds = (df['System_Time'].iloc[-1] - df['System_Time'].iloc[0]).total_seconds()
if total_time_seconds > 0:
    fs = len(df) / total_time_seconds
else:
    fs = 50.0  # Default value
    
print(f"Estimated sampling frequency (fs) for current data is approx: {fs:.2f} Hz")

cutoff = CUTOFF_FREQ  # Cutoff frequency
order = FILTER_ORDER  # Filter order

nyq = 0.5 * fs
if cutoff >= nyq:
    print(f"Warning: The set cutoff frequency ({cutoff}Hz) is greater than or equal to the Nyquist frequency ({nyq:.2f}Hz), skipping filtering!")
else:
    # Design Butterworth filter
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter to each column
    for col in filter_cols:
        # Fill missing values to prevent filtfilt errors (forward fill then backward fill)
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill()
        
        # Use filtfilt for zero-phase filtering to prevent signal lag and phase shift
        df[col] = filtfilt(b, a, df[col])
        
print("Filtering completed!")

# ---------------------------------------------------------
# 3. Mark valid data segments (for subsequent Z-score calculation)
# ---------------------------------------------------------
print("Marking valid data segments...")
# Default all data as valid task data
df['Is_Valid_Task'] = 1

# --- Process redundant data at the beginning and end (based on Sync_Mark labels) ---
if 'Sync_Mark' in df.columns:
    # Look for SYNC_START_BOARD label (robust processing)
    start_marks = df[df['Sync_Mark'].astype(str).str.contains('SYNC_START_BOARD', na=False, case=False)]
    if not start_marks.empty:
        # Get the time of the first appearance of the start label
        exp_start_time = start_marks['System_Time'].iloc[0]
        # Mark all data before this as invalid (0)
        df.loc[df['System_Time'] < exp_start_time, 'Is_Valid_Task'] = 0
        print(f"Experiment start label recognized, successfully removed preparation data before {exp_start_time}.")

    # Look for SYNC_END_BOARD label
    end_marks = df[df['Sync_Mark'].astype(str).str.contains('SYNC_END_BOARD', na=False, case=False)]
    if not end_marks.empty:
        # Get the time of the last appearance of the end label
        exp_end_time = end_marks['System_Time'].iloc[-1]
        # Mark all data after this as invalid (0)
        df.loc[df['System_Time'] > exp_end_time, 'Is_Valid_Task'] = 0
        print(f"Experiment end label recognized, successfully removed concluding data after {exp_end_time}.")
else:
    print("Tip: 'Sync_Mark' column not found in data, full data will be used for subsequent calculations.")

# Note: The hardcoded interference time period removal logic for old datasets has been deleted to accommodate general new datasets.

# ---------------------------------------------------------
# 4. Z-score Normalization (calculate mean and std using valid data only)
# ---------------------------------------------------------
print("Performing Z-score normalization (calculating parameters based on valid task segments only)...")

# Create a mask containing only valid data to calculate pure μ and σ
valid_mask = df['Is_Valid_Task'] == 1

for col in filter_cols:
    # Calculate mean and standard deviation only from valid data to prevent environmental noise or preparation movements from polluting the overall distribution
    clean_mean = df.loc[valid_mask, col].mean()
    clean_std = df.loc[valid_mask, col].std()
    
    if clean_std != 0 and pd.notna(clean_std):
        # Standardize the entire column using the calculated distribution parameters
        df[col] = (df[col] - clean_mean) / clean_std

print("Normalization completed!")

# ---------------------------------------------------------
# 5. Save processed data and add timestamp
# ---------------------------------------------------------
# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Extract original filename
original_filename = os.path.splitext(os.path.basename(file_path))[0]
base_dir = os.path.dirname(file_path)

# Specify target save path (Creates a 'processed_data' folder in the same directory as the input file)
save_dir = os.path.join(base_dir, "processed_data")

# If the folder does not exist, the program will create it automatically
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
        print(f"Target folder created: {save_dir}")
    except Exception as e:
        print(f"Failed to create folder, will save to the current working directory. Reason: {e}")
        save_dir = "."

# Concatenate new filename
save_filename = f"{original_filename}_processed_{timestamp}.csv"
save_path = os.path.join(save_dir, save_filename)

# Export to CSV file
try:
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"Processing completed! New file successfully saved to: {save_path}")
except Exception as e:
    print(f"Error saving file: {e}")
# ---------------------------------------------------------