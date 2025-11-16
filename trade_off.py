import matplotlib 
matplotlib.use('Agg') # Đề phòng lỗi RuntimeError

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# (Cần cài đặt: pip install xgboost)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

# --- CÁC THAM SỐ CỐ ĐỊNH ---
INPUT_FILE = "rfm_training_data.csv" 
FEATURE_COLUMNS = ['Recency', 'Frequency', 'Monetary']
TARGET_COLUMN = 'y_HighValueChurn' 
POSITIVE_CLASS_NAME = 'High Value Churn (1)' 
NEGATIVE_CLASS_NAME = 'Khác (0)'
MIN_LEAF_SIZE = 10 
N_TREES_RF = 500
N_TREES_XG = 100
DEPTH_RANGE = range(3, 16) # Quét độ sâu từ 3 đến 15

print("\n" + "="*50)
print(f"--- BẮT ĐẦU CHẨN ĐOÁN CÁC MÔ HÌNH CÂY ---")
print("="*50 + "\n")

# --- 1. TẢI VÀ CHIA DỮ LIỆU (CHỈ 1 LẦN) ---
try:
    final_df = pd.read_csv(INPUT_FILE)
    print(f"--- 1. Đã tải file '{INPUT_FILE}' thành công ---")
except FileNotFoundError:
    print(f"!!! LỖI: Không tìm thấy file '{INPUT_FILE}'...")
    exit()

X = final_df[FEATURE_COLUMNS]
y = final_df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    
    random_state=42,  
    stratify=y        
)
print("\n--- 2. Đã chia dữ liệu thành Train (80%) và Test (20%) ---")

# --- 2. TÍNH scale_pos_weight (Cho XGBoost) ---
counts = y_train.value_counts()
scale_pos_weight = counts[0] / counts[1]
print(f"   Tính toán 'scale_pos_weight' (cho XGB) = {scale_pos_weight:.2f}")


# --- 3. HÀM VẼ BIỂU ĐỒ (Dùng chung) ---
def plot_metrics(depths, f1, prec, rec, acc, model_name):
    """Hàm này vẽ 2 biểu đồ so sánh 4 chỉ số theo max_depth"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Biểu đồ 1: P, R, F1
    ax1.plot(depths, f1, 'r-o', label='F1-Score (Test)')
    ax1.plot(depths, prec, 'b--o', label='Precision (Test)')
    ax1.plot(depths, rec, 'g--o', label='Recall (Test)')
    ax1.set_title(f'So sánh F1, Precision, Recall ({model_name})', fontsize=14)
    ax1.set_ylabel('Điểm (0.0 - 1.0)', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ 2: Accuracy
    ax2.plot(depths, acc, 'k-o', label='Accuracy (Test - Tổng thể)')
    ax2.set_title('Chỉ số Accuracy (Tổng thể)', fontsize=14)
    ax2.set_xlabel('max_depth', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    output_file_all = f"churn_{model_name.lower()}_all_metrics_vs_depth.png"
    plt.savefig(output_file_all)
    plt.close() 
    print(f"\nĐã lưu biểu đồ so sánh cho {model_name} vào file: {output_file_all}")

# --- 4. CHẠY VÀ VẼ CHO TỪNG MÔ HÌNH ---

# === 4.1. DECISION TREE ===
print(f"\n--- 3. Đang quét max_depth cho Decision Tree... ---")
metrics_dt = {'f1': [], 'prec': [], 'rec': [], 'acc': []}

for depth in DEPTH_RANGE:
    model_temp = DecisionTreeClassifier(
        class_weight='balanced', 
        max_depth=depth, 
        min_samples_leaf=MIN_LEAF_SIZE, 
        criterion='gini',      
        random_state=42, 
    )
    model_temp.fit(X_train, y_train)
    y_pred_test = model_temp.predict(X_test)
    
    metrics_dt['f1'].append(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_dt['prec'].append(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_dt['rec'].append(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_dt['acc'].append(accuracy_score(y_test, y_pred_test))

print("Quét Decision Tree xong.")
plot_metrics(DEPTH_RANGE, metrics_dt['f1'], metrics_dt['prec'], metrics_dt['rec'], metrics_dt['acc'], "DecisionTree")

# === 4.2. RANDOM FOREST ===
print(f"\n--- 4. Đang quét max_depth cho Random Forest... ---")
metrics_rf = {'f1': [], 'prec': [], 'rec': [], 'acc': []}

for depth in DEPTH_RANGE:
    model_temp = RandomForestClassifier(
        n_estimators=N_TREES_RF,
        n_jobs=-1,
        class_weight='balanced', 
        max_depth=depth, 
        criterion='gini',      
        random_state=42, 
    )
    model_temp.fit(X_train, y_train)
    y_pred_test = model_temp.predict(X_test)
    
    metrics_rf['f1'].append(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_rf['prec'].append(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_rf['rec'].append(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_rf['acc'].append(accuracy_score(y_test, y_pred_test))

print("Quét Random Forest xong.")
plot_metrics(DEPTH_RANGE, metrics_rf['f1'], metrics_rf['prec'], metrics_rf['rec'], metrics_rf['acc'], "RandomForest")

# === 4.3. XGBOOST ===
print(f"\n--- 5. Đang quét max_depth cho XGBoost... ---")
metrics_xgb = {'f1': [], 'prec': [], 'rec': [], 'acc': []}

for depth in DEPTH_RANGE:
    model_temp = XGBClassifier(
        n_estimators=N_TREES_XG,
        max_depth=depth, 
        random_state=42,
        scale_pos_weight=scale_pos_weight, # Dùng tham số riêng của XGB
        n_jobs=-1,
        eval_metric='logloss'
    )
    model_temp.fit(X_train, y_train)
    y_pred_test = model_temp.predict(X_test)
    
    metrics_xgb['f1'].append(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_xgb['prec'].append(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_xgb['rec'].append(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_xgb['acc'].append(accuracy_score(y_test, y_pred_test))

print("Quét XGBoost xong.")
plot_metrics(DEPTH_RANGE, metrics_xgb['f1'], metrics_xgb['prec'], metrics_xgb['rec'], metrics_xgb['acc'], "XGBoost")


print("\n" + "="*50)
print("--- CHẨN ĐOÁN HOÀN TẤT (3 file ảnh đã được lưu) ---")
print("="*50 + "\n")