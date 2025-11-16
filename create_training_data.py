import pandas as pd
import datetime as dt

# --- 1. Thiết lập ---
INPUT_FILE = 'Online Retail.xlsx'
OUTPUT_FILE = 'rfm_training_data.csv'
SNAPSHOT_DATE = dt.datetime(2011, 10, 9)
HIGH_VALUE_MONETARY_QUANTILE = 0.33 
HIGH_VALUE_FREQ_QUANTILE = 0.25   
print(f"--- BẮT ĐẦU QUÁ TRÌNH TẠO DỮ LIỆU HUẤN LUYỆN ---")
print(f"File đầu vào: {INPUT_FILE}")
print(f"Mốc thời gian (Snapshot Date): {SNAPSHOT_DATE.date()}")
print(f"File đầu ra: {OUTPUT_FILE}")

# --- 2. Tiền xử lý ---
try:
    df = pd.read_excel(INPUT_FILE)
    print("\n--- 2. Tải dữ liệu gốc thành công ---")
except FileNotFoundError:
    print(f"\nLỖI: Không tìm thấy file {INPUT_FILE}.")
    exit()
    
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.dropna(subset=['CustomerID'], inplace=True) 
df = df[df['Quantity'] > 0]                 
df = df[df['UnitPrice'] > 0]                
df = df[df['Country'] == 'United Kingdom']  

# Tính TotalCost
df['TotalCost'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int)

# --- 3. Phân chia dữ liệu ---
past_df = df[df['InvoiceDate'] < SNAPSHOT_DATE].copy()
future_df = df[df['InvoiceDate'] >= SNAPSHOT_DATE].copy()
print(f"Đã chia dữ liệu tại mốc {SNAPSHOT_DATE.date()}:")
print(f"  Giao dịch 'Quá khứ' để tính RFM: {len(past_df)} dòng")
print(f"  Giao dịch 'Tương lai' để tạo nhãn: {len(future_df)} dòng")

# --- 4. Tính toán RFM feature X ở quá khứ ---
print("\n--- 4. Bắt đầu tính toán R, F, M (features) ---")
rfm_features_df = past_df.groupby('CustomerID').agg(
    Last_Purchase_Date=pd.NamedAgg(column='InvoiceDate', aggfunc='max'),
    Frequency=pd.NamedAgg(column='InvoiceNo', aggfunc='nunique'),
    Monetary=pd.NamedAgg(column='TotalCost', aggfunc='sum')
).reset_index()
rfm_features_df['Monetary'] = rfm_features_df['Monetary'].round(1).astype(int)

# Tính Recency
rfm_features_df['Recency'] = (SNAPSHOT_DATE - rfm_features_df['Last_Purchase_Date']).dt.days
rfm_features_df.drop(columns=['Last_Purchase_Date'], inplace=True)
print("Tính toán R, F, M hoàn tất.")

# --- 5. Tạo biến mục tiêu ---
print("\n--- 5. Bắt đầu tạo biến mục tiêu 'y_HighValueChurn' ---")
p_monetary = rfm_features_df['Monetary'].quantile(HIGH_VALUE_MONETARY_QUANTILE)
p_frequency = rfm_features_df['Frequency'].quantile(HIGH_VALUE_FREQ_QUANTILE)
returned_customers = set(future_df['CustomerID'].unique())
print(f"Tìm thấy {len(returned_customers)} khách hàng có quay lại mua sau mốc {SNAPSHOT_DATE.date()}.")

# Hàm tạo nhãn
def define_target_variable(row):
    is_high_value_M = row['Monetary'] >= p_monetary
    is_high_value_F = row['Frequency'] >= p_frequency  
    is_high_value = is_high_value_M and is_high_value_F
    is_churned = row['CustomerID'] not in returned_customers
    
    if is_high_value and is_churned:
        return 1
    else:
        return 0
    
rfm_features_df['y_HighValueChurn'] = rfm_features_df.apply(define_target_variable, axis=1)
print("Đã gán nhãn 'y_HighValueChurn' cho tất cả khách hàng.")

# --- 6. Lưu file ---
print("\n--- 6. Hoàn tất. Kiểm tra kết quả ---")
print(f"Thống kê biến mục tiêu (y):")
print(rfm_features_df['y_HighValueChurn'].value_counts())

# Kiểm tra 5 dòng đầu
final_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'y_HighValueChurn']
final_df = rfm_features_df[final_columns]
print(f"\n5 dòng đầu của file dữ liệu huấn luyện:")
print(final_df.head())
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nĐã lưu file huấn luyện hoàn chỉnh vào: {OUTPUT_FILE}")
print("--- QUÁ TRÌNH KẾT THÚC ---")