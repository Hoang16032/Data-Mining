import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 0. THIẾT LẬP ---
FILE_NAME = "rfm_training_data.csv"
sns.set(style='whitegrid')
plt.rcParams['font.family'] = 'Arial'

print(f"--- Đọc file: {FILE_NAME} ---")
rfm_df = pd.read_csv(FILE_NAME)

# --- BƯỚC 1: TÍNH TOÁN ĐIỂM R, F, M (Giống hệt script trước) ---
print("--- 1. Bắt đầu gán điểm R, F, M ---")
quantiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile([.25, .5, .75]).to_dict()

# Hàm gán điểm R (1=Tốt, 4=Tệ)
def r_score(x, q_dict):
    q = q_dict['Recency']
    if x <= q[.25]: return 1
    elif x <= q[.5]: return 2
    elif x <= q[.75]: return 3
    else: return 4

# Hàm gán điểm F, M (1=Tốt, 4=Tệ)
def fm_score(x, col, q_dict):
    q = q_dict[col]
    if x <= q[.25]: return 4
    elif x <= q[.5]: return 3
    elif x <= q[.75]: return 2
    else: return 1

rfm_df['R_Score'] = rfm_df['Recency'].apply(r_score, args=(quantiles,))
rfm_df['F_Score'] = rfm_df['Frequency'].apply(fm_score, args=('Frequency', quantiles))
rfm_df['M_Score'] = rfm_df['Monetary'].apply(fm_score, args=('Monetary', quantiles))
print("Gán điểm R, F, M hoàn tất.")


# --- BƯỚC 2: PHÂN KHÚC 4 NHÓM CHÍNH ---
print("--- 2. Phân khúc 4 nhóm chính (High Risk, Low Risk...) ---")

def rfm_segment_v2(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

    # 1. High Risk (Nguy cơ cao)
    # Lâu không mua (R=3,4) NHƯNG F/M tốt (F/M=1,2)
    if (r in [3, 4]) and (f in [1, 2]) and (m in [1, 2]):
        return 'High Risk'

    # 2. Low Risk (Nguy cơ thấp)
    # Lâu không mua (R=3,4) VÀ F/M cũng tệ (F/M=3,4)
    elif (r in [3, 4]) and (f in [3, 4]) and (m in [3, 4]):
        return 'Low Risk'

    # 3. Champions (Khách VIP)
    # Mọi thứ đều tốt (R/F/M = 1)
    elif r == 1 and f == 1 and m == 1:
        return 'Champions (VIP)'
    
    # 4. Other (Các nhóm còn lại: Loyal, New, Promising...)
    else:
        return 'Other (Khác)'

rfm_df['Segment'] = rfm_df.apply(rfm_segment_v2, axis=1)


# --- BƯỚC 3: TÍNH TOÁN GIÁ TRỊ CỦA CÁC NHÓM RỦI RO ---
print("--- 3. Thống kê tổng giá trị của các nhóm ---")

# Tính tổng số lượng KH và tổng Doanh thu cho mỗi nhóm
segment_analysis = rfm_df.groupby('Segment').agg(
    CustomerCount=('CustomerID', 'count'),
    TotalMonetary=('Monetary', 'sum')
).reset_index()

print(segment_analysis)

# Lọc ra 2 nhóm rủi ro để so sánh
risk_df = segment_analysis[segment_analysis['Segment'].isin(['High Risk', 'Low Risk'])].copy()
# Tính %
risk_df['%_Count'] = (risk_df['CustomerCount'] / risk_df['CustomerCount'].sum()) * 100
risk_df['%_Monetary'] = (risk_df['TotalMonetary'] / risk_df['TotalMonetary'].sum()) * 100

print("\n--- Phân tích 2 nhóm rủi ro (High vs Low) ---")
print(risk_df)

# --- BƯỚC 4: TRỰC QUAN HÓA CÂU CHUYỆN ---
print("--- 4. Bắt đầu vẽ biểu đồ và lưu file ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 4.1. Biểu đồ 1: % SỐ LƯỢNG KHÁCH HÀNG
colors = ['#FF6B6B', '#4ECDC4'] # Đỏ cho High Risk, Xanh cho Low Risk
ax1.pie(
    risk_df['%_Count'],
    labels=risk_df['Segment'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
ax1.set_title('Biểu đồ 1: Phân bổ SỐ LƯỢNG\ntrong Nhóm Rủi ro', fontsize=16, fontweight='bold')

# 4.2. Biểu đồ 2: % TỔNG GIÁ TRỊ DOANH THU
ax2.pie(
    risk_df['%_Monetary'],
    labels=risk_df['Segment'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
ax2.set_title('Biểu đồ 2: Phân bổ TỔNG GIÁ TRỊ\ntrong Nhóm Rủi ro', fontsize=16, fontweight='bold')

plt.suptitle('So sánh High Risk vs. Low Risk', fontsize=20, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('risk_comparison.png')

print("Đã lưu biểu đồ so sánh vào file 'plot_risk_comparison.png'")
print("--- KẾT THÚC BƯỚC 2 ---")