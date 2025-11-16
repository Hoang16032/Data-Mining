import pandas as pd
import datetime as dt

# --- 1. THIẾT LẬP CÁC BIẾN QUAN TRỌNG ---
INPUT_FILE = 'Online Retail.xlsx'
OUTPUT_FILE = 'rfm_training_data.csv'

# Mốc thời gian "hiện tại" để tính RFM (features X)
SNAPSHOT_DATE = dt.datetime(2011, 10, 9)

# Ngưỡng phân vị để xác định "High Value"
HIGH_VALUE_MONETARY_QUANTILE = 0.33 # THAY ĐỔI: Ngưỡng cho Monetary (P50)
HIGH_VALUE_FREQ_QUANTILE = 0.25    # THAY ĐỔI: Thêm ngưỡng cho Frequency (P25)

print(f"--- BẮT ĐẦU QUÁ TRÌNH TẠO DỮ LIỆU HUẤN LUYỆN ---")
print(f"File đầu vào: {INPUT_FILE}")
print(f"Mốc thời gian (Snapshot Date): {SNAPSHOT_DATE.date()}")
print(f"File đầu ra: {OUTPUT_FILE}")

# --- 2. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU GỐC ---
try:
    df = pd.read_excel(INPUT_FILE)
    print("\n--- 2. Tải dữ liệu gốc thành công ---")
except FileNotFoundError:
    print(f"\nLỖI: Không tìm thấy file {INPUT_FILE}.")
    exit()

# Chuyển InvoiceDate sang kiểu datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 2.1. Áp dụng logic làm sạch từ file PDF của bạn
df.dropna(subset=['CustomerID'], inplace=True) # Bỏ null CustomerID
df = df[df['Quantity'] > 0]                 # Bỏ trả hàng/lỗi
df = df[df['UnitPrice'] > 0]                # Bỏ trả hàng/lỗi
df = df[df['Country'] == 'United Kingdom']  # Chỉ tập trung vào UK

# 2.2. Tính TotalCost
df['TotalCost'] = df['Quantity'] * df['UnitPrice']

# Chuyển CustomerID sang kiểu int cho sạch
df['CustomerID'] = df['CustomerID'].astype(int)

print("Đã làm sạch dữ liệu: bỏ null, lọc UK, lọc giao dịch lỗi.")

# --- 3. PHÂN CHIA DỮ LIỆU "QUÁ KHỨ" VÀ "TƯƠNG LAI" ---
# "Quá khứ": Dùng để tính features (R, F, M)
past_df = df[df['InvoiceDate'] < SNAPSHOT_DATE].copy()

# "Tương lai": Dùng để quan sát và tạo nhãn (y)
future_df = df[df['InvoiceDate'] >= SNAPSHOT_DATE].copy()

print(f"Đã chia dữ liệu tại mốc {SNAPSHOT_DATE.date()}:")
print(f"  Giao dịch 'Quá khứ' (để tính RFM): {len(past_df)} dòng")
print(f"  Giao dịch 'Tương lai' (để tạo nhãn): {len(future_df)} dòng")

# --- 4. TÍNH TOÁN RFM FEATURES (X) TỪ "QUÁ KHỨ" ---
print("\n--- 4. Bắt đầu tính toán R, F, M (features) ---")

rfm_features_df = past_df.groupby('CustomerID').agg(
    Last_Purchase_Date=pd.NamedAgg(column='InvoiceDate', aggfunc='max'),
    Frequency=pd.NamedAgg(column='InvoiceNo', aggfunc='nunique'),
    Monetary=pd.NamedAgg(column='TotalCost', aggfunc='sum')
).reset_index()
rfm_features_df['Monetary'] = rfm_features_df['Monetary'].round(1).astype(int)

# 4.1. Tính Recency
rfm_features_df['Recency'] = (SNAPSHOT_DATE - rfm_features_df['Last_Purchase_Date']).dt.days
rfm_features_df.drop(columns=['Last_Purchase_Date'], inplace=True)

print("Tính toán R, F, M hoàn tất.")

# --- 5. TẠO BIẾN MỤC TIÊU (y) TỪ "TƯƠNG LAI" ---
print("\n--- 5. Bắt đầu tạo biến mục tiêu 'y_HighValueChurn' ---")

# --- THAY ĐỔI: Cập nhật bước 5.1 ---
# 5.1. Xác định các ngưỡng "High Value"
# Ngưỡng này được tính từ chính dữ liệu quá khứ
p_monetary = rfm_features_df['Monetary'].quantile(HIGH_VALUE_MONETARY_QUANTILE)
p_frequency = rfm_features_df['Frequency'].quantile(HIGH_VALUE_FREQ_QUANTILE)

print(f"Ngưỡng High Value (Monetary >= P{int(HIGH_VALUE_MONETARY_QUANTILE*100)}) là: {p_monetary:.2f}")
print(f"Ngưỡng High Value (Frequency >= P{int(HIGH_VALUE_FREQ_QUANTILE*100)}) là: {p_frequency:.2f}")


# 5.2. Tìm danh sách khách hàng có quay lại trong "tương lai"
returned_customers = set(future_df['CustomerID'].unique())
print(f"Tìm thấy {len(returned_customers)} khách hàng có quay lại mua sau mốc {SNAPSHOT_DATE.date()}.")


# --- THAY ĐỔI: Cập nhật bước 5.3 ---
# 5.3. Định nghĩa hàm tạo nhãn
def define_target_variable(row):
    # Điều kiện 1: Khách hàng có phải High Value không?
    # PHẢI THỎA MÃN CẢ 2 ĐIỀU KIỆN M VÀ F
    is_high_value_M = row['Monetary'] >= p_monetary
    is_high_value_F = row['Frequency'] >= p_frequency
    
    is_high_value = is_high_value_M and is_high_value_F # Đây là thay đổi logic chính
    
    # Điều kiện 2: Khách hàng có rời bỏ (không quay lại) không?
    is_churned = row['CustomerID'] not in returned_customers
    
    # Nếu thỏa mãn CẢ HAI điều kiện (High Value VÀ Churned)
    if is_high_value and is_churned:
        return 1
    else:
        return 0

# 5.4. Áp dụng hàm để tạo cột 'y'
rfm_features_df['y_HighValueChurn'] = rfm_features_df.apply(define_target_variable, axis=1)
print("Đã gán nhãn 'y_HighValueChurn' (0 hoặc 1) cho tất cả khách hàng.")

# --- 6. KIỂM TRA VÀ LƯU FILE ---
print("\n--- 6. Hoàn tất. Kiểm tra kết quả ---")
print(f"Thống kê biến mục tiêu (y):")
print(rfm_features_df['y_HighValueChurn'].value_counts())

# 6.1. Chỉ lưu các cột cần thiết cho mô hình
final_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'y_HighValueChurn']
final_df = rfm_features_df[final_columns]

print(f"\n5 dòng đầu của file dữ liệu huấn luyện:")
print(final_df.head())

# 6.2. Lưu file
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nĐã lưu file huấn luyện hoàn chỉnh vào: {OUTPUT_FILE}")
print("--- QUÁ TRÌNH KẾT THÚC ---")