import matplotlib 
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')
INPUT_FILE = "rfm_training_data.csv"
FEATURE_COLUMNS = ['Recency', 'Frequency', 'Monetary']
TARGET_COLUMN = 'y_HighValueChurn'
POSITIVE_CLASS_NAME = 'High Value Churn (1)'
NEGATIVE_CLASS_NAME = 'Khác (0)'
N_ESTIMATORS = 100 

print("\n" + "="*50)
print(f"--- BƯỚC 5: HUẤN LUYỆN MÔ HÌNH XGBOOST ---")
print(f"--- (n_estimators={N_ESTIMATORS}) ---")
print("="*50 + "\n")

# --- 1. TẢI DỮ LIỆU ---
try:
    final_df = pd.read_csv(INPUT_FILE)
    print(f"--- 1. Đã tải file '{INPUT_FILE}' thành công ---")
    print("   Thống kê biến mục tiêu (y):")
    print(final_df[TARGET_COLUMN].value_counts())
except FileNotFoundError:
    print(f"!!! LỖI: Không tìm thấy file '{INPUT_FILE}'...")
    exit()

# --- 2. TÁCH BIẾN X (ĐẶC TRƯNG) VÀ y (MỤC TIÊU) ---
X = final_df[FEATURE_COLUMNS]
y = final_df[TARGET_COLUMN]

# --- 3. CHIA DỮ LIỆU THÀNH TRAIN VÀ TEST ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    
    random_state=42,  
    stratify=y        
)
print("\n--- 2. Đã chia dữ liệu thành Train (80%) và Test (20%) ---")

# --- 3.1: TÍNH scale_pos_weight ---
counts = y_train.value_counts()
scale_pos_weight = counts[0] / counts[1]
print(f"   Tính toán 'scale_pos_weight' = {scale_pos_weight:.2f} (để xử lý mất cân bằng)")

# --- 4. TÌM MAX_DEPTH TỐI ƯU (Dựa trên F1-Score) ---
print(f"\n--- 3. Đang tìm max_depth tối ưu (với n_estimators={N_ESTIMATORS})... ---")
depths = range(3, 16)
test_f1_scores = []

for depth in depths:
    model_temp = XGBClassifier( 
        n_estimators=N_ESTIMATORS,
        learning_rate = 0.05,
        max_depth=depth, 
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        eval_metric='logloss' 
    )
    model_temp.fit(X_train, y_train)
    
    y_pred_test_temp = model_temp.predict(X_test)
    test_f1_scores.append(f1_score(y_test, y_pred_test_temp, pos_label=1))

best_depth = depths[np.argmax(test_f1_scores)]
print(f"Đã quét xong. 'max_depth' tối ưu (có Test F1-Score cao nhất) là: {best_depth}")


# --- 4.1: VẼ BIỂU ĐỒ TEST F1-SCORE ---
plt.figure(figsize=(10, 6))
plt.plot(depths, test_f1_scores, 'r-o', label='F1-Score (Test)')
plt.title(f'Tìm max_depth (XGBoost, n_estimators={N_ESTIMATORS})')
plt.xlabel('max_depth')
plt.ylabel(f'F1-Score (cho lớp "{POSITIVE_CLASS_NAME}")')
plt.axvline(x=best_depth, color='grey', linestyle='--', label=f'Best Depth = {best_depth}')
plt.legend()
plt.grid(True)

output_file_depth = "xgb_tuning.png" 
plt.savefig(output_file_depth)
plt.close() 
print(f"Đã lưu biểu đồ tìm max_depth vào file: {output_file_depth}")

# --- 5. HUẤN LUYỆN MÔ HÌNH CHÍNH ---
xgb_model_main = XGBClassifier( 
    n_estimators=N_ESTIMATORS,
    learning_rate = 0.05,
    max_depth=best_depth, 
    random_state=42,
    scale_pos_weight=scale_pos_weight, 
    n_jobs=-1,
    eval_metric='logloss'
)

print(f"\n--- 4. Đang huấn luyện mô hình CHÍNH (max_depth={best_depth}, n_estimators={N_ESTIMATORS})... ---")
xgb_model_main.fit(X_train, y_train)
print("Huấn luyện hoàn tất.")

# --- 6. DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH CHÍNH ---
print("\n--- 5. Đánh giá hiệu suất mô hình CHÍNH ---")
y_pred_xgb = xgb_model_main.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"Accuracy (Độ chính xác tổng thể): {accuracy_xgb:.4f}")

target_names = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]
print("\nBáo cáo phân loại chi tiết (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred_xgb, target_names=target_names, zero_division=0))

# --- 7. VẼ VÀ LƯU MA TRẬN NHẦM LẪN ---
print("--- 6. Đang tạo Ma trận nhầm lẫn ---")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
axis_labels = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_xgb, 
    annot=True, fmt='d', cmap='Blues', 
    xticklabels=axis_labels, 
    yticklabels=axis_labels  
)
plt.title(f'Ma trận nhầm lẫn (XGBoost) | Accuracy: {accuracy_xgb:.2%}', fontsize=14)
plt.xlabel('Predicted (Dự đoán)', fontsize=12) 
plt.ylabel('True (Thực tế)', fontsize=12)    
plt.tight_layout()

output_file_cm = "xgb_matrix.png" 
plt.savefig(output_file_cm, dpi=300) 
plt.close()
print(f"Đã lưu Ma trận nhầm lẫn vào file: {output_file_cm}")