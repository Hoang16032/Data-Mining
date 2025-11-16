import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
FILE_NAME = "rfm_training_data.csv"
sns.set(style='whitegrid')

# --- Bước 1: Tải dữ liệu từ datamart ---
try:
    rfm_df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {FILE_NAME}. Vui lòng chạy Script 1 trước.")
    exit()
print("--- 1. Đọc dữ liệu RFM (từ file rfm_training_data.csv) ---")
print(rfm_df.head())
print("\n" + "="*50 + "\n")

# --- Bước 2: Tính các cột mốc phân vị ---
quantiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile([.25, .5, .75]).to_dict()
print("--- 2. Các mốc phân vị (Quantiles) ---")
print(pd.DataFrame(quantiles))
print("\n" + "="*50 + "\n")

# --- Bước 3: Xây dựng hàn gán điểm (Scoring) ---
# 3.1. Hàm gán điểm cho Recency (R)
def r_score(x, quantiles_dict):
    q = quantiles_dict['Recency']
    if x <= q[.25]:
        return 1 
    elif x <= q[.5]:
        return 2
    elif x <= q[.75]:
        return 3
    else:
        return 4 

# 3.2. Hàm gán điểm cho Frequency (F) và Monetary (M)
def fm_score(x, col_name, quantiles_dict):
    q = quantiles_dict[col_name]
    if x <= q[.25]:
        return 4 
    elif x <= q[.5]:
        return 3
    elif x <= q[.75]:
        return 2
    else:
        return 1 

print("--- 3. Bắt đầu gán điểm cho R, F, M ---")
rfm_df['R_Score'] = rfm_df['Recency'].apply(r_score, args=(quantiles,))
rfm_df['F_Score'] = rfm_df['Frequency'].apply(fm_score, args=('Frequency', quantiles))
rfm_df['M_Score'] = rfm_df['Monetary'].apply(fm_score, args=('Monetary', quantiles))
print("Đã gán điểm R, F, M xong.")
print("\n" + "="*50 + "\n")

# --- Bước 4: Tạo điểm RFM tổng hợp ---
rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + \
                      rfm_df['F_Score'].astype(str) + \
                      rfm_df['M_Score'].astype(str)
print("--- 4. Đã kết hợp điểm RFM_Score ---")
print(rfm_df[['CustomerID', 'RFM_Score']].head())
print("\n" + "="*50 + "\n")

# --- BƯỚC 5: GÁN NHÃN PHÂN KHÚC  ---
def rfm_segment(row):
    r = row['R_Score']
    f = row['F_Score']
    m = row['M_Score']

    # Champions (Tốt nhất: Mới mua, mua thường xuyên, chi nhiều tiền nhất)
    if (r in [1, 2]) and f == 1 and m == 1: return 'Champions (Khách VIP)'
    # High Risk (VIP cũ đã lâu không quay lại)
    elif (r in [3, 4]) and (f in [1, 2]) and (m in [1, 3]): return 'High Risk (Có nguy cơ rời bỏ & GT cao)'
    # Low Risk (Lâu không mua, tần suất và chi tiêu thấp)
    elif (r in [3, 4]) and (f in [3, 4]) and (m in [3, 4]): return 'Low Risk (Có nguy cơ rời bỏ & GT thấp)'
    # Promising (Mới mua, nhưng F/M thấp nhất. Là một nhóm 'Khách mới' cần quan tâm)
    elif r == 1 and f == 4 and m == 4: return 'Promising (Khách hứa hẹn)'
    # Loyal Customers (Trung thành: Mua thường xuyên & chi tiêu tốt. Đặt sau Champions)
    elif r == 2 and (f in [1, 2]) and (m in [1, 2]): return 'Loyal Customers (Khách trung thành)'
    # Potential Loyalist (Tiềm năng: Gần đây, mua/chi tiêu khá tốt. Đặt sau Loyal)
    elif (r in [1, 2]) and (f in [1, 3]) and (m in [1, 3]): return 'Potential Loyalist (Tiềm năng)'
    # Customers Needing Attention (Cần chú ý: Các chỉ số ở mức trung bình, có thể đang chững lại)
    elif (r in [2, 3]) and (f in [2, 3]) and (m in [2, 3]): return 'Customers Needing Attention (Cần chú ý)'
    else: return 'Other (Khác)'

rfm_df['Segment'] = rfm_df.apply(rfm_segment, axis=1)
print("--- 5. Đã gán nhãn 9 phân khúc (đã gộp High Risk) ---")
print(rfm_df[['CustomerID', 'RFM_Score', 'Segment']].head())
print("\n" + "="*50 + "\n")

# --- BƯỚC 6: XEM KẾT QUẢ CUỐI CÙNG VÀ THỐNG KÊ ---
print("--- 6. Thống kê số lượng khách hàng theo phân khúc ---")
segment_counts = rfm_df['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'CustomerCount']
print(segment_counts)

print("\n--- 6.1. Thống kê RFM trung bình của mỗi phân khúc (để giải thích) ---")
segment_stats = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
segment_stats_sorted = segment_stats.sort_values(by='Monetary', ascending=False).reset_index(drop=True)
print(segment_stats_sorted)
print("\n" + "="*50 + "\n")

# --- BƯỚC 7: TRỰC QUAN HÓA DỮ LIỆU ---
print("--- 7. Bắt đầu vẽ biểu đồ và lưu file ---")

# 7.1. BIỂU ĐỒ TRÒN: Tỷ lệ số lượng khách hàng 
plt.figure(figsize=(13, 9))
plt.pie(
    segment_counts['CustomerCount'],
    labels=segment_counts['Segment'],
    autopct='%1.1f%%', # Hiển thị %
    startangle=90,
    pctdistance=0.85,
    colors=sns.color_palette('Paired')
)

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Biểu đồ 1: Phân bổ số lượng khách hàng theo phân khúc', fontsize=16)
plt.tight_layout()
plt.savefig('segment_distribution.png', dpi=300)
print("Đã lưu Biểu đồ 1 (Donut Chart) vào file 'segment_distribution.png'")


# 7.2. BIỂU ĐỒ CỘT: Giá trị trung bình của mỗi phân khúc 
plt.figure(figsize=(13, 8))
barplot = sns.barplot(
    data=segment_stats_sorted,
    y='Segment',
    x='Monetary',
    palette='viridis'
)
plt.title('Biểu đồ 2: Giá trị chi tiêu trung bình (Monetary) của mỗi phân khúc', fontsize=16)
plt.xlabel('Chi tiêu trung bình (Monetary)', fontsize=12)
plt.ylabel('Phân khúc khách hàng', fontsize=12)
barplot.bar_label(barplot.containers[0], fmt='%.2f', padding=5)
plt.tight_layout()
plt.savefig('segment_value.png', dpi=300)
print("Đã lưu Biểu đồ 2 (Bar Chart) vào file 'segment_value.png'")

print("\n--- KẾT THÚC BƯỚC 2 ---")