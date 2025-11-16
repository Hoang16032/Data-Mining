import pandas as pd
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

# --- 1. THIẾT LẬP CÁC BIẾN QUAN TRỌNG ---
INPUT_FILE = 'Online Retail.xlsx'
OUTPUT_FILE = 'rfm_datamart.csv' 
    
SNAPSHOT_DATE = dt.datetime(2011, 10, 9) 

print(f"--- BẮT ĐẦU QUÁ TRÌNH TẠO FILE RFM DATAMART ---")
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

# (Giữ nguyên phần làm sạch)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.dropna(subset=['CustomerID'], inplace=True) 
df = df[df['Quantity'] > 0]                 
df = df[df['UnitPrice'] > 0]                
df = df[df['Country'] == 'United Kingdom']  
df['TotalCost'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int)
print("Đã làm sạch dữ liệu.")

# --- 3. LỌC DỮ LIỆU "QUÁ KHỨ" ---
past_df = df[df['InvoiceDate'] < SNAPSHOT_DATE].copy()
print(f"Đã lọc {len(past_df)} dòng giao dịch trước mốc {SNAPSHOT_DATE.date()}.")

# --- 4. TÍNH TOÁN R, F, M (FEATURES) ---
print("\n--- 4. Bắt đầu tính toán R, F, M (features) ---")
    
rfm_datamart = past_df.groupby('CustomerID').agg(
    Last_Purchase_Date=pd.NamedAgg(column='InvoiceDate', aggfunc='max'),
    Frequency=pd.NamedAgg(column='InvoiceNo', aggfunc='nunique'),
    Monetary=pd.NamedAgg(column='TotalCost', aggfunc='sum')
).reset_index()
rfm_datamart['Monetary'] = rfm_datamart['Monetary'].round(1).astype(int)

# 4.1. Tính Recency
rfm_datamart['Recency'] = (SNAPSHOT_DATE - rfm_datamart['Last_Purchase_Date']).dt.days
rfm_datamart.drop(columns=['Last_Purchase_Date'], inplace=True) # Xóa cột tạm
    
print("Tính toán R, F, M hoàn tất.")

# --- 5. LƯU FILE ---
final_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
final_df = rfm_datamart[final_columns]

print("\n--- 5. Hoàn tất. Kiểm tra kết quả ---")
print(f"\n5 dòng đầu của file dữ liệu RFM (Monetary đã làm tròn):")
print(final_df.head())

# 5.2. Lưu file
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nĐã lưu file RFM Datamart hoàn chỉnh vào: {OUTPUT_FILE}")
print("--- QUÁ TRÌNH KẾT THÚC ---")