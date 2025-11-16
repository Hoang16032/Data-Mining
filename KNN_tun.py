import matplotlib 
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler   
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
K_RANGE = range(3, 26, 2) 

print("\n" + "="*50)
print(f"--- BẮT ĐẦU CHẨN ĐOÁN MÔ HÌNH KNN ---")
print("="*50 + "\n")

# --- 1. TẢI VÀ CHIA DỮ LIỆU (CHỈ 1 LẦN) ---
try:
    final_df = pd.read_csv(INPUT_FILE)
    print(f"--- 1. Đã tải file '{INPUT_FILE}' thành công ---")
except FileNotFoundError:
    print(f"!!! LỖI: Không tìm thấy file '{INPUT_FILE}'...")
    print("Vui lòng chạy Script 1 (create_training_data.py) trước.")
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

# --- 3. CHUẨN HÓA DỮ LIỆU  ---
print("\n--- 3. Đang chuẩn hóa (StandardScaler) dữ liệu... ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
print("Chuẩn hóa hoàn tất.")

# --- 4. HÀM VẼ BIỂU ĐỒ ---
def plot_metrics_knn(k_values, f1, prec, rec, acc, model_name):
    """Hàm này vẽ 2 biểu đồ so sánh 4 chỉ số theo k"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # Biểu đồ 1: P, R, F1
    ax1.plot(k_values, f1, 'r-o', label='F1-Score (Test)')
    ax1.plot(k_values, prec, 'b--o', label='Precision (Test)')
    ax1.plot(k_values, rec, 'g--o', label='Recall (Test)')
    ax1.set_title(f'So sánh F1, Precision, Recall ({model_name})', fontsize=14)
    ax1.set_ylabel('Điểm (0.0 - 1.0)', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ 2: Accuracy
    ax2.plot(k_values, acc, 'k-o', label='Accuracy (Test - Tổng thể)')
    ax2.set_title('Chỉ số Accuracy (Tổng thể)', fontsize=14)
    ax2.set_xlabel('k (n_neighbors)', fontsize=12) 
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    output_file_all = f"optimize_{model_name.lower()}.png"
    plt.savefig(output_file_all)
    plt.close() 
    print(f"\nĐã lưu biểu đồ so sánh cho {model_name} vào file: {output_file_all}")

# --- 5. CHẠY VÀ VẼ CHO KNN ---
print(f"\n--- 4. Đang quét 'k' (n_neighbors) cho KNN... ---")
metrics_knn = {'f1': [], 'prec': [], 'rec': [], 'acc': []}

for k in K_RANGE:
    model_temp = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1,
        weights='distance' 
    )
    model_temp.fit(X_train_scaled, y_train)
    y_pred_test = model_temp.predict(X_test_scaled)
    
    metrics_knn['f1'].append(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_knn['prec'].append(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_knn['rec'].append(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    metrics_knn['acc'].append(accuracy_score(y_test, y_pred_test))

print("Quét KNN xong.")
plot_metrics_knn(K_RANGE, metrics_knn['f1'], metrics_knn['prec'], metrics_knn['rec'], metrics_knn['acc'], "KNN")

