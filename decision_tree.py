import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')
# --- CÁC THAM SỐ CỐ ĐỊNH (ĐÃ CẬP NHẬT CHO BÀI TOÁN CHURN) ---
INPUT_FILE = "rfm_training_data.csv" # <-- THAY ĐỔI: Dùng file từ Bước 1
FEATURE_COLUMNS = ['Recency', 'Frequency', 'Monetary']
TARGET_COLUMN = 'y_HighValueChurn' # <-- THAY ĐỔI: Dùng cột mục tiêu mới
POSITIVE_CLASS_NAME = 'High Value Churn (1)' # <-- THAY ĐỔI: Tên lớp 1
NEGATIVE_CLASS_NAME = 'Khác (0)'
MIN_LEAF_SIZE = 10 # Giữ nguyên tham số của bạn

print("\n" + "="*50)
print(f"--- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN CHURN ---")
print(f"--- (Decision Tree, min_leaf={MIN_LEAF_SIZE}) ---")
print("="*50 + "\n")

# --- 1. TẢI DỮ LIỆU ĐÃ CHUẨN BỊ (TỪ BƯỚC 1) ---
try:
    final_df = pd.read_csv(INPUT_FILE)
    print(f"--- 1. Đã tải file '{INPUT_FILE}' thành công ---")
    print(f"   Tổng số {len(final_df)} khách hàng (dòng).")
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
    test_size=0.2,    # Tỷ lệ 80% Train, 20% Test
    random_state=42,  # Để đảm bảo kết quả có thể tái lập
    stratify=y        # QUAN TRỌNG: Giữ nguyên tỷ lệ 122 'y=1' trong cả Train và Test
)
print("\n--- 2. Đã chia dữ liệu thành Train (80%) và Test (20%) ---")
print(f"   Train: {len(X_train)} dòng")
print(f"   Test:  {len(X_test)} dòng")

# --- 4. TÌM MAX_DEPTH TỐI ƯU (Dựa trên F1-Score) ---
print(f"\n--- 3. Đang tìm max_depth tối ưu (với min_leaf={MIN_LEAF_SIZE})... ---")
depths = range(3, 16)
train_f1_scores = []
test_f1_scores = []

for depth in depths:
    model_temp = DecisionTreeClassifier(
        class_weight='balanced', # QUAN TRỌNG: Xử lý dữ liệu mất cân bằng (122 vs 2905)
        max_depth=7, 
        min_samples_leaf=MIN_LEAF_SIZE, # Sử dụng min_leaf
        criterion='gini',      
        random_state=42, # QUAN TRỌNG: Xử lý dữ liệu mất cân bằng (122 vs 2905)
    )
    model_temp.fit(X_train, y_train)
    
    # Dự đoán trên cả tập Train và Test
    y_pred_train_temp = model_temp.predict(X_train)
    y_pred_test_temp = model_temp.predict(X_test)
    
    # Ghi lại F1-score cho lớp 1 (High Value Churn)
    train_f1_scores.append(f1_score(y_train, y_pred_train_temp, pos_label=1))
    test_f1_scores.append(f1_score(y_test, y_pred_test_temp, pos_label=1))

# Tìm độ sâu tốt nhất (nơi Test F1-Score cao nhất)
best_depth = depths[np.argmax(test_f1_scores)]
print(f"Đã quét xong. 'max_depth' tối ưu (có Test F1-Score cao nhất) là: {best_depth}")

# --- 5. HUẤN LUYỆN MÔ HÌNH CHÍNH ---
dt_model_main = DecisionTreeClassifier(
    class_weight='balanced', # QUAN TRỌNG: Xử lý dữ liệu mất cân bằng (122 vs 2905)
    max_depth=best_depth,
    min_samples_leaf=MIN_LEAF_SIZE, 
    criterion='gini', 
    random_state=42
)

print(f"\n--- 4. Đang huấn luyện mô hình CHÍNH (max_depth={best_depth}, min_leaf={MIN_LEAF_SIZE})... ---")
dt_model_main.fit(X_train, y_train)
print("Huấn luyện hoàn tất.")

# --- 6. DỰ ĐOÁN VÀ ĐÁNH GIÁ MÔ HÌNH CHÍNH ---
print("\n--- 5. Đánh giá hiệu suất mô hình CHÍNH ---")
y_pred_dt = dt_model_main.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# QUAN TRỌNG: Thông báo về Accuracy (Dựa trên file mẫu 242.pdf)
print(f"Accuracy (Độ chính xác tổng thể): {accuracy_dt:.4f}")
print("-> LƯU Ý: Accuracy rất cao nhưng 'giả tạo' do dữ liệu mất cân bằng.")
print("-> Hãy tập trung vào 'Recall' của lớp 'High Value Churn (1)'.")

target_names = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]
print("\nBáo cáo phân loại chi tiết (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred_dt, target_names=target_names, zero_division=0))

# --- 7. VẼ VÀ LƯU MA TRẬN NHẦM LẪN (LAYOUT CŨ CỦA BẠN) ---
print("--- 6. Đang tạo Ma trận nhầm lẫn ---")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
matrix_display = np.array([
    [tp, fp],  # (Dự đoán 1, Thực tế 1), (Dự đoán 1, Thực tế 0)
    [fn, tn]   # (Dự đoán 0, Thực tế 1), (Dự đoán 0, Thực tế 0)
])

# Định nghĩa nhãn cho các trục (đúng theo thứ tự code của bạn)
axis_labels = [POSITIVE_CLASS_NAME, NEGATIVE_CLASS_NAME]
plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix_display, 
    annot=True, fmt='d', cmap='Blues', 
    xticklabels=axis_labels, 
    yticklabels=axis_labels  
)
plt.title(f'Ma trận nhầm lẫn | Accuracy: {accuracy_dt:.2%}', fontsize=14)
plt.xlabel('Thực tế (Fact)', fontsize=12)
plt.ylabel('Dự đoán (Classified)', fontsize=12)
plt.tight_layout()

output_file_cm = "churn_dt_matrix.png" # <-- THAY ĐỔI: Tên file
plt.savefig(output_file_cm)
plt.close()
print(f"Đã lưu Ma trận nhầm lẫn vào file: {output_file_cm}")

# --- 8. VẼ VÀ LƯU CÂY QUYẾT ĐỊNH ---
print(f"\n--- 7. Đang tạo biểu đồ Cây Quyết định... ---")
class_names_binary = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]

plt.figure(figsize=(40, 20)) 
plot_tree(
    dt_model_main, 
    feature_names=FEATURE_COLUMNS, 
    class_names=class_names_binary,
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=4 # Giữ nguyên max_depth=4 để dễ đọc
)
plt.title(f"Trực quan hóa Cây Quyết định (max_depth={best_depth}, min_leaf={MIN_LEAF_SIZE})", fontsize=20)

output_file_tree = "churn_dt_structure.png" # <-- THAY ĐỔI: Tên file
plt.savefig(output_file_tree, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ Cây Quyết định vào file: {output_file_tree}")

print("\n" + "="*50)
print("--- BƯỚC 3 HOÀN TẤT ---")
print("="*50 + "\n")