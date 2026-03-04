import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc
)
import xgboost as xgb

"""
Machine Learning Model Training - Chronological Block Split
This script performs:
1. Data loading and temporal sorting.
2. Chronological Block Splitting: Data is divided into N-minute blocks.
3. Interleaved Assignment: Even blocks for training, odd blocks for testing.
4. Model training (XGBoost) and performance visualization.
"""

# ==========================================
# CONFIGURATION
# ==========================================
BLOCK_SIZE_MINUTES = 5  # Duration of each time block in minutes
RANDOM_STATE = 42
XGB_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
# ==========================================

# ---------------------------------------------------------
# 1. Select the feature dataset
# ---------------------------------------------------------
root = tk.Tk()
root.withdraw()

print("Step 1: Please select the generated feature dataset (e.g., ML_Features_Dataset_xxx.csv)...")
file_path = filedialog.askopenfilename(
    title="Select ML Feature Dataset",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    print("No file selected, program terminated.")
    sys.exit()

print(f"Loading data: {file_path}")
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# 2. Preprocessing & Chronological Block Splitting
# ---------------------------------------------------------
if df.isnull().sum().sum() > 0:
    print("Missing values detected, imputing with median values...")
    df = df.fillna(df.median(numeric_only=True))

# Ensure Time_Stamp is datetime and sorted
df['Time_Stamp'] = pd.to_datetime(df['Time_Stamp'])
df = df.sort_values('Time_Stamp').reset_index(drop=True)

# Calculate Time Blocks
start_time = df['Time_Stamp'].min()
df['Elapsed_Minutes'] = (df['Time_Stamp'] - start_time).dt.total_seconds() / 60.0
# Calculate Block_ID (0 for 1st block, 1 for 2nd block, etc.)
df['Block_ID'] = (df['Elapsed_Minutes'] // BLOCK_SIZE_MINUTES).astype(int)

# Interleaved Assignment:
# Training Set: Blocks 1, 3, 5, 7... (Block_ID % 2 == 0)
# Testing Set: Blocks 2, 4, 6, 8... (Block_ID % 2 != 0)
train_df = df[df['Block_ID'] % 2 == 0].copy()
test_df = df[df['Block_ID'] % 2 != 0].copy()

def get_X_y(data):
    """Helper to drop utility columns and return features and labels"""
    drop_cols = ['Time_Stamp', 'y_Label', 'Elapsed_Minutes', 'Block_ID']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    y = data['y_Label']
    return X, y

X_train, y_train = get_X_y(train_df)
X_test, y_test = get_X_y(test_df)

print(f"\n>>> Data Splitting Complete ({BLOCK_SIZE_MINUTES}-min Block Interleaving):")
print(f"Training: Block IDs {[i+1 for i in sorted(train_df['Block_ID'].unique())]}, Samples={len(X_train)}")
print(f"Testing: Block IDs {[i+1 for i in sorted(test_df['Block_ID'].unique())]}, Samples={len(X_test)}")
print(f"Label distribution (Test Set):\n{y_test.value_counts(normalize=True)}")

# Calculate class weights for imbalance
scale_pos_weight_val = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1

# ---------------------------------------------------------
# 3. Prepare Directory for Saving Results
# ---------------------------------------------------------
now_str = datetime.now().strftime("%m%d_%H%M")
base_dir = os.path.dirname(file_path)
target_dir = os.path.join(base_dir, "block_split_results", f"Run_{now_str}")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"\nResults directory created: {target_dir}")

# ---------------------------------------------------------
# 4. Train XGBoost Model
# ---------------------------------------------------------
print("\n>>> Starting model training...")
xgb_model = xgb.XGBClassifier(
    n_estimators=XGB_ESTIMATORS, 
    learning_rate=XGB_LEARNING_RATE, 
    scale_pos_weight=scale_pos_weight_val, 
    random_state=RANDOM_STATE,
    eval_metric='aucpr',
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. Result Visualization & Image Saving
# ---------------------------------------------------------

def save_report_as_img(y_true, y_pred, model_name, path):
    report = classification_report(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.text(0.01, 0.05, str(report), {'fontsize': 12, 'fontfamily': 'monospace'})
    plt.axis('off')
    plt.title(f'{model_name} - Interleaved Block Test Report', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_curves(model, X_test, y_test, save_dir):
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Time-Block Splitting)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'ROC_Curve.png'), dpi=300)
    plt.close()

    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve (Time-Block Splitting)')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'PR_Curve.png'), dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('XGBoost Feature Importance (Top 20)', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Execute visualization
print("Generating analysis charts...")
y_pred_xgb = xgb_model.predict(X_test)

save_report_as_img(y_test, y_pred_xgb, "XGBoost", os.path.join(target_dir, 'Classification_Report.png'))
plot_curves(xgb_model, X_test, y_test, target_dir)
plot_feature_importance(xgb_model, X_train.columns, os.path.join(target_dir, 'Feature_Importance_Analysis.png'))

print(f"\nEvaluation complete! Results saved to:\n{target_dir}")