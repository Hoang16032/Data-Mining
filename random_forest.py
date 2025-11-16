import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier # <-- THAY ĐỔI: Dùng Random Forest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import os
import matplotlib 
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# --- CÁC THAM SỐ CỐ ĐỊNH (ĐÃ CẬP NHẬT CHO BÀI TOÁN CHURN) ---
INPUT_FILE = "rfm_training_data.csv"
FEATURE_COLUMNS = ['Recency', 'Frequency', 'Monetary']
TARGET_COLUMN = 'y_HighValueChurn'
POSITIVE_CLASS_NAME = 'High Value Churn (1)'
NEGATIVE_CLASS_NAME = 'Khác (0)'
N_TREES = 500 # Số lượng cây trong rừng


print("\n" + "="*50)
print(f"--- BƯỚC 4: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ---")
print(f"--- (n_estimators={N_TREES}) ---")
print("="*50 + "\n")

# --- 1. TẢI DỮ LIỆU ĐÃ CHUẨN BỊ (TỪ BƯỚC 1) ---
try:
    final_df = pd.read_csv(INPUT_FILE)
    print(f"--- 1. Đã tải file '{INPUT_FILE}' thành công ---")
    print("   Thống kê biến mục tiêu (y):")
    print(final_df[TARGET_COLUMN].value_counts())
except FileNotFoundError:
    print(f"!!! LỖI: Không tìm thấy file '{INPUT_FILE}'...")
    print("Vui lòng chạy Script 1 (create_training_data.py) trước.")
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
print(f"   Train: {len(X_train)} dòng")
print(f"   Test:  {len(X_test)} dòng")

# --- 4. TÌM MAX_DEPTH TỐI ƯU (Dựa trên F1-Score) ---
print(f"\n--- 3. Đang tìm max_depth tối ưu (với n_trees={N_TREES})... ---")
depths = range(3, 16)
train_f1_scores = []
test_f1_scores = []

for depth in depths:
    model_temp = RandomForestClassifier( # <-- THAY ĐỔI
        n_estimators=N_TREES,    # Số cây
        max_depth=depth, 
        criterion='gini',      
        random_state=42,
        class_weight='balanced', # Rất quan trọng
        n_jobs=-1                # Dùng tất cả CPU
    )
    model_temp.fit(X_train, y_train)
    
    y_pred_train_temp = model_temp.predict(X_train)
    y_pred_test_temp = model_temp.predict(X_test)
    
    train_f1_scores.append(f1_score(y_train, y_pred_train_temp, pos_label=1))
    test_f1_scores.append(f1_score(y_test, y_pred_test_temp, pos_label=1))

best_depth = depths[np.argmax(test_f1_scores)]
print(f"Đã quét xong. 'max_depth' tối ưu (có Test F1-Score cao nhất) là: {best_depth}")


# --- 4.1: VẼ BIỂU ĐỒ TEST/TRAIN F1-SCORE ---
plt.figure(figsize=(10, 6))
plt.plot(depths, train_f1_scores, 'b-o', label='F1-Score (Train)')
plt.plot(depths, test_f1_scores, 'r-o', label='F1-Score (Test)')
plt.title(f'Tìm max_depth (Random Forest, n_trees={N_TREES})')
plt.xlabel('max_depth')
plt.ylabel(f'F1-Score (cho lớp "{POSITIVE_CLASS_NAME}")')
plt.axvline(x=best_depth, color='grey', linestyle='--', label=f'Best Depth = {best_depth}')
plt.legend()
plt.grid(True)

output_file_depth = "churn_rf_tuning.png" # <-- THAY ĐỔI
plt.savefig(output_file_depth)
plt.close() 
print(f"Đã lưu biểu đồ tìm max_depth vào file: {output_file_depth}")

# --- 5. HUẤN LUYỆN MÔ HÌNH CHÍNH ---
rf_model_main = RandomForestClassifier( # <-- THAY ĐỔI
    n_estimators=N_TREES,
    max_depth=best_depth, 
    criterion='gini', 
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print(f"\n--- 4. Đang huấn luyện mô hình CHÍNH (max_depth={best_depth}, n_trees={N_TREES})... ---")
rf_model_main.fit(X_train, y_train)
print("Huấn luyện hoàn tất.")

# --- 6. DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH CHÍNH ---
print("\n--- 5. Đánh giá hiệu suất mô hình CHÍNH ---")
y_pred_rf = rf_model_main.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy (Độ chính xác tổng thể): {accuracy_rf:.4f}")
print("-> LƯU Ý: Tập trung vào 'Recall' của lớp 'High Value Churn (1)'.")

target_names = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]
print("\nBáo cáo phân loại chi tiết (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred_rf, target_names=target_names, zero_division=0))

# --- 7. VẼ VÀ LƯU MA TRẬN NHẦM LẪN ---
print("--- 6. Đang tạo Ma trận nhầm lẫn ---")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
matrix_display = np.array([
    [tp, fp],  
    [fn, tn]   
])

axis_labels = [POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME]
plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix_display, 
    annot=True, fmt='d', cmap='Blues', 
    xticklabels=axis_labels, 
    yticklabels=axis_labels  
)
plt.title(f'Ma trận nhầm lẫn (Random Forest) | Accuracy: {accuracy_rf:.2%}', fontsize=14)
plt.xlabel('Thực tế (Fact)', fontsize=12)
plt.ylabel('Dự đoán (Classified)', fontsize=12)
plt.tight_layout()

output_file_cm = "churn_rf_matrix.png" # <-- THAY ĐỔI
plt.savefig(output_file_cm)
plt.close()
print(f"Đã lưu Ma trận nhầm lẫn vào file: {output_file_cm}")

# --- 8. VẼ VÀ LƯU MỨC ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG ---
print(f"\n--- 7. Đang tạo biểu đồ Feature Importance... ---")

# Lấy thông tin từ mô hình đã huấn luyện
importances = rf_model_main.feature_importances_
feature_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=feature_df, 
    x='Importance', 
    y='Feature', 
    palette='viridis'
)
plt.title('Mức độ quan trọng của Đặc trưng (Feature Importance)', fontsize=16)
plt.xlabel('Mức độ quan trọng', fontsize=12)
plt.ylabel('Đặc trưng', fontsize=12)
barplot.bar_label(barplot.containers[0], fmt='%.3f', padding=5) # Thêm số %
plt.tight_layout()

output_file_features = "churn_rf_feature_importance.png" # <-- THAY ĐỔI
plt.savefig(output_file_features, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ Feature Importance vào file: {output_file_features}")

print("\n" + "="*50)
print("--- BƯỚC 4 HOÀN TẤT ---")
print("="*50 + "\n")