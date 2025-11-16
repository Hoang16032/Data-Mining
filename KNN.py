import matplotlib 
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler  
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

# --- 4. CHUẨN HÓA DỮ LIỆU (SCALING) ---
# Đây là bước BẮT BUỘC cho KNN
print("\n--- 3. Đang chuẩn hóa (StandardScaler) dữ liệu... ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
print("Chuẩn hóa hoàn tất.")

# --- 5. TÌM k TỐI ƯU (n_neighbors) ---
print(f"\n--- 4. Đang tìm 'k' (n_neighbors) tối ưu... ---")
k_values = range(3, 22, 2) 
test_f1_scores = []

for k in k_values:
    model_temp = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1,
        weights='distance' 
    )
    model_temp.fit(X_train_scaled, y_train)
    y_pred_test_temp = model_temp.predict(X_test_scaled)
    test_f1_scores.append(f1_score(y_test, y_pred_test_temp, pos_label=1))

# Tìm k tốt nhất (nơi F1-Score cao nhất)
best_k = k_values[np.argmax(test_f1_scores)]
print(f"Đã quét xong. 'k' tối ưu (có Test F1-Score cao nhất) là: {best_k}")

# --- 5.1: VẼ BIỂU ĐỒ TEST F1-SCORE ---
plt.figure(figsize=(10, 6))
plt.plot(k_values, test_f1_scores, 'r-o', label='F1-Score (Test)')
plt.title(f'Tìm k (n_neighbors) tối ưu cho KNN')
plt.xlabel('k (Số hàng xóm)')
plt.ylabel(f'F1-Score (cho lớp "{POSITIVE_CLASS_NAME}")')
plt.axvline(x=best_k, color='grey', linestyle='--', label=f'Best k = {best_k}')
plt.legend()
plt.grid(True)
output_file_depth = "knn_tuning.png"
plt.savefig(output_file_depth)
plt.close() 
print(f"Đã lưu biểu đồ tìm 'k' vào file: {output_file_depth}")

# --- 6. HUẤN LUYỆN MÔ HÌNH CHÍNH ---
knn_model_main = KNeighborsClassifier(
    n_neighbors=best_k,
    n_jobs=-1,
    weights='distance'
)

print(f"\n--- 5. Đang huấn luyện mô hình CHÍNH (k={best_k})... ---")
knn_model_main.fit(X_train_scaled, y_train) 
print("Huấn luyện hoàn tất.")

# --- 7. DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH CHÍNH ---
print("\n--- 6. Đánh giá hiệu suất mô hình ---")
y_pred_knn = knn_model_main.predict(X_test_scaled) 
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy: {accuracy_knn:.4f}")

target_names = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]
print("\nBáo cáo phân loại chi tiết (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred_knn, target_names=target_names, zero_division=0))

# --- 8. VẼ VÀ LƯU MA TRẬN NHẦM LẪN ---
print("--- 7. Đang tạo Ma trận nhầm lẫn ---")
cm_knn = confusion_matrix(y_test, y_pred_knn)
axis_labels = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_knn, 
    annot=True, fmt='d', cmap='Blues', 
    xticklabels=axis_labels, 
    yticklabels=axis_labels  
)
plt.title(f'Ma trận nhầm lẫn (KNN) | Accuracy: {accuracy_knn:.2%}', fontsize=14)
plt.xlabel('Predicted (Dự đoán)', fontsize=12) 
plt.ylabel('True (Thực tế)', fontsize=12)   
plt.tight_layout()

output_file_cm = "knn_matrix.png" 
plt.savefig(output_file_cm, dpi=300) 
plt.close()
print(f"Đã lưu Ma trận nhầm lẫn vào file: {output_file_cm}")