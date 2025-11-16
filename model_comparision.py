import matplotlib 
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

model_results = [
    {
        'Model': 'Decision Tree', 
        'Accuracy': 0.7672,  
        'Precision': 0.56,   
        'Recall': 0.92,      
        'F1-Score': 0.69      
    },
    {
        'Model': 'Random Forest', 
        'Accuracy': 0.7731,  
        'Precision': 0.56,   
        'Recall': 0.93,      
        'F1-Score': 0.70      
    },
    {
        'Model': 'XGBoost', 
        'Accuracy': 0.7746,  
        'Precision': 0.57,   
        'Recall': 0.91,      
        'F1-Score': 0.70      
    },
    {
        'Model': 'KNN', 
        'Accuracy': 0.7209,  
        'Precision': 0.51,   
        'Recall': 0.47,      
        'F1-Score': 0.49     
    }
]
# --- 2. TẠO DATAFRAME TỪ KẾT QUẢ ---
df_results = pd.DataFrame(model_results)
df_results.set_index('Model', inplace=True)

print("--- 1. Đã tải dữ liệu kết quả ---")
print(df_results)

# --- 3. VẼ BIỂU ĐỒ SO SÁNH  ---
print("\n--- 2. Đang vẽ biểu đồ tổng kết 2x2... ---")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('So sánh Hiệu suất 4 mô hình trên lớp "High Value Churn"', fontsize=16, y=1.03)
sns.set(style='whitegrid')
df_sorted = df_results.sort_values(by='F1-Score', ascending=False)
def add_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                    fontweight='bold')

# Biểu đồ 1: Accuracy
sns.barplot(data=df_sorted, x=df_sorted.index, y='Accuracy', ax=axes[0, 0], palette='Blues_d')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_ylabel('Điểm (0.0 - 1.0)')
axes[0, 0].set_ylim(0, 1.05)
add_labels(axes[0, 0])

# Biểu đồ 2: Precision
sns.barplot(data=df_sorted, x=df_sorted.index, y='Precision', ax=axes[0, 1], palette='Oranges_d')
axes[0, 1].set_title('Precision')
axes[0, 1].set_ylabel('Điểm (0.0 - 1.0)')
axes[0, 1].set_ylim(0, 1.05)
add_labels(axes[0, 1])

# Biểu đồ 3: Recall (Quan trọng nhất)
sns.barplot(data=df_sorted, x=df_sorted.index, y='Recall', ax=axes[1, 0], palette='Greens_d')
axes[1, 0].set_title('Recall')
axes[1, 0].set_ylabel('Điểm (0.0 - 1.0)')
axes[1, 0].set_ylim(0, 1.05)
add_labels(axes[1, 0])

# Biểu đồ 4: F1-Score
sns.barplot(data=df_sorted, x=df_sorted.index, y='F1-Score', ax=axes[1, 1], palette='Reds_d')
axes[1, 1].set_title('F1-Score')
axes[1, 1].set_ylabel('Điểm (0.0 - 1.0)')
axes[1, 1].set_ylim(0, 1.05)
add_labels(axes[1, 1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
output_file_all = "model_comparison.png"
plt.savefig(output_file_all, dpi=300, bbox_inches='tight')
plt.close() 

print(f"\nĐã lưu biểu đồ so sánh 2x2 vào file: {output_file_all}")
print("--- HOÀN TẤT ---")