import matplotlib 
matplotlib.use('Agg') # Đề phòng lỗi RuntimeError

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*50)
print(f"--- BƯỚC 8: VẼ BIỂU ĐỒ TỔNG KẾT (DẠNG 2x2 GRID) ---")
print("="*50 + "\n")

# --- 1. NHẬP KẾT QUẢ CỦA BẠN VÀO ĐÂY ---
#
# HÃY CHẠY 4 SCRIPT (DT, RF, XGB, KNN) VÀ ĐIỀN KẾT QUẢ VÀO DANH SÁCH DƯỚI ĐÂY
# Lấy các chỉ số 'Precision', 'Recall', 'f1-score' cho lớp 'High Value Churn (1)'
# Lấy 'Accuracy' là chỉ số tổng thể (ví dụ: 0.9094)
#
model_results = [
    {
        'Model': 'Decision Tree', 
        'Accuracy': 0.9094, # (22+530)/(22+52+3+530)
        'Precision': 0.297,  # 22 / (22 + 52)
        'Recall': 0.880,     # 22 / (22 + 3)
        'F1-Score': 0.444     # F1 của lớp (1)
    },
    {
        'Model': 'Random Forest', 
        'Accuracy': 0.0, # <-- ĐIỀN KẾT QUẢ TỪ SCRIPT 4 VÀO ĐÂY
        'Precision': 0.0, 
        'Recall': 0.0, 
        'F1-Score': 0.0
    },
    {
        'Model': 'XGBoost', 
        'Accuracy': 0.0, # <-- ĐIỀN KẾT QUẢ TỪ SCRIPT 5 VÀO ĐÂY
        'Precision': 0.0, 
        'Recall': 0.0, 
        'F1-Score': 0.0
    },
    {
        'Model': 'KNN', 
        'Accuracy': 0.0, # <-- ĐIỀN KẾT QUẢ TỪ SCRIPT 7 VÀO ĐÂY
        'Precision': 0.0, 
        'Recall': 0.0, 
        'F1-Score': 0.0
    }
]
# --- (Kết thúc phần nhập liệu) ---


# --- 2. TẠO DATAFRAME TỪ KẾT QUẢ ---
df_results = pd.DataFrame(model_results)
df_results.set_index('Model', inplace=True)

print("--- 1. Đã tải dữ liệu kết quả ---")
print(df_results)

# --- 3. VẼ BIỂU ĐỒ SO SÁNH (Giống file 242.pdf) ---
print("\n--- 2. Đang vẽ biểu đồ tổng kết 2x2... ---")

fig, axes = plt.subplots(2, 2, figsize=(15, 12)) # Tạo 4 ô (2x2)
fig.suptitle('So sánh Hiệu suất 4 Mô hình (trên Lớp "High Value Churn")', fontsize=16, y=1.03)
sns.set(style='whitegrid')

# Sắp xếp theo F1-Score để biểu đồ đẹp hơn
df_sorted = df_results.sort_values(by='F1-Score', ascending=False)

# Hàm phụ để thêm số lên trên cột
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                    fontweight='bold')

# Biểu đồ 1: Accuracy
sns.barplot(data=df_sorted, x=df_sorted.index, y='Accuracy', ax=axes[0, 0], palette='Blues_d')
axes[0, 0].set_title('So sánh Accuracy (Tổng thể)')
axes[0, 0].set_ylabel('Điểm (0.0 - 1.0)')
axes[0, 0].set_ylim(0, 1.05)
add_labels(axes[0, 0])

# Biểu đồ 2: Precision
sns.barplot(data=df_sorted, x=df_sorted.index, y='Precision', ax=axes[0, 1], palette='Oranges_d')
axes[0, 1].set_title('So sánh Precision (Lớp "Churn")')
axes[0, 1].set_ylabel('Điểm (0.0 - 1.0)')
axes[0, 1].set_ylim(0, 1.05)
add_labels(axes[0, 1])

# Biểu đồ 3: Recall (Quan trọng nhất)
sns.barplot(data=df_sorted, x=df_sorted.index, y='Recall', ax=axes[1, 0], palette='Greens_d')
axes[1, 0].set_title('So sánh Recall (Lớp "Churn") - CHỈ SỐ QUAN TRỌNG NHẤT')
axes[1, 0].set_ylabel('Điểm (0.0 - 1.0)')
axes[1, 0].set_ylim(0, 1.05)
add_labels(axes[1, 0])

# Biểu đồ 4: F1-Score
sns.barplot(data=df_sorted, x=df_sorted.index, y='F1-Score', ax=axes[1, 1], palette='Reds_d')
axes[1, 1].set_title('So sánh F1-Score (Lớp "Churn")')
axes[1, 1].set_ylabel('Điểm (0.0 - 1.0)')
axes[1, 1].set_ylim(0, 1.05)
add_labels(axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh layout
output_file_all = "churn_model_comparison_GRID.png"
plt.savefig(output_file_all, dpi=300, bbox_inches='tight')
plt.close() 

print(f"\nĐã lưu biểu đồ so sánh 2x2 vào file: {output_file_all}")
print("--- HOÀN TẤT ---")