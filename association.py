import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --- LOGIC TÍNH ĐIỂM (TỪ BƯỚC 3 CỦA SCRIPT 2) ---
# Chúng ta định nghĩa các hàm này ở ngoài để sử dụng bên trong hàm chính

def r_score(x, quantiles_dict):
    """Gán điểm cho Recency (R). Số nhỏ (mới) tốt hơn (điểm 1)."""
    q = quantiles_dict['Recency']
    if x <= q[.25]:
        return 1 # Tốt nhất
    elif x <= q[.5]:
        return 2
    elif x <= q[.75]:
        return 3
    else:
        return 4 # Tệ nhất

def fm_score(x, col_name, quantiles_dict):
    """Gán điểm cho Frequency (F) và Monetary (M). Số lớn tốt hơn (điểm 1)."""
    q = quantiles_dict[col_name]
    if x <= q[.25]:
        return 4 # Tệ nhất
    elif x <= q[.5]:
        return 3
    elif x <= q[.75]:
        return 2
    else:
        return 1 # Tốt nhất

# -----------------------------------------------------------------

def run_association_rules_on_rfm(rfm_df, min_support=0.01, min_confidence=0.5, min_lift=1.0):
    """
    Hàm này thực hiện toàn bộ quy trình:
    1. Rời rạc hóa (Discretize) dữ liệu RFM *theo PHÂN VỊ (Quantiles)*.
    2. Chuyển đổi (One-Hot Encode) thành định dạng giỏ hàng.
    3. Chạy Apriori để tìm các tập mục phổ biến (frequent itemsets).
    4. Tạo luật kết hợp (association rules).
    5. Lọc ra các luật mạnh (Strong Rules) dựa trên Lift.
    """
    
    # --- 1. Rời rạc hóa (Discretize) THEO LOGIC MỚI (TỪ SCRIPT 2) ---
    print("--- 1. Bắt đầu rời rạc hóa dữ liệu RFM (theo Phân vị) ---")

    # --- Tương đương BƯỚC 2 (Script 2): TÍNH CÁC MỐC PHÂN VỊ (QUANTILES) ---
    try:
        quantiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile([.25, .5, .75]).to_dict()
        print("Các mốc phân vị (Quantiles) được sử dụng:")
        print(pd.DataFrame(quantiles))
    except Exception as e:
        print(f"LỖI khi tính phân vị: {e}. Dữ liệu có thể quá nhỏ hoặc có vấn đề.")
        return

    # --- Tương đương BƯỚC 3 (Script 2): GÁN ĐIỂM (SCORING) ---
    rfm_df['R_Score'] = rfm_df['Recency'].apply(r_score, args=(quantiles,))
    rfm_df['F_Score'] = rfm_df['Frequency'].apply(fm_score, args=('Frequency', quantiles))
    rfm_df['M_Score'] = rfm_df['Monetary'].apply(fm_score, args=('Monetary', quantiles))
    
    print("\nDữ liệu RFM sau khi gán điểm (Scoring):")
    print(rfm_df[['CustomerID', 'R_Score', 'F_Score', 'M_Score']].head())
    print("-" * 40)


    # --- 2. Chuẩn bị dữ liệu cho Apriori (One-Hot Encoding) ---
    print("--- 2. Chuyển đổi sang định dạng 'giỏ hàng' (One-Hot) ---")
    
    # Chỉ lấy các cột điểm số
    # CHÚ Ý: Chuyển sang kiểu 'str' để get_dummies tạo ra các cột 
    # R_Score=1, R_Score=2, F_Score=1... 
    rfm_scores_str = rfm_df[['R_Score', 'F_Score', 'M_Score']].astype(str)

    # Bỏ qua các hàng có giá trị NaN (nếu có)
    rfm_scores_str = rfm_scores_str.dropna()

    # Chuyển đổi thành DataFrame one-hot
    # Dùng prefix_sep='=' để có tên cột đẹp: R_Score=1, F_Score=4, ...
    basket_df = pd.get_dummies(rfm_scores_str, prefix_sep='=')
    
    # Chuyển giá trị 0/1 thành False/True (mlxtend chấp nhận cả hai)
    basket_df = basket_df.astype(bool)

    print("Dữ liệu One-Hot (đầu vào cho Apriori):")
    print(basket_df.head())
    print("-" * 40)

    # --- 3. Chạy Apriori để tìm Tập mục phổ biến (Frequent Itemsets) ---
    print(f"--- 3. Chạy Apriori (min_support = {min_support}) ---")
    
    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        print(f"KHÔNG TÌM THẤY TẬP MỤC PHỔ BIẾN. Hãy thử giảm 'min_support'.")
        return

    print("Các tập mục phổ biến đã tìm thấy:")
    print(frequent_itemsets.sort_values(by='support', ascending=False).head())
    print("-" * 40)

    # --- 4. Tạo Luật kết hợp (Association Rules) ---
    print(f"--- 4. Tạo Luật (min_confidence = {min_confidence}) ---")
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        print(f"KHÔNG TÌM THẤY LUẬT. Hãy thử giảm 'min_confidence' hoặc 'min_support'.")
        return
        
    print(f"Tổng số luật tìm thấy (trước khi lọc Lift): {len(rules)}")
    print("-" * 40)

    # --- 5. Lọc ra các Luật Mạnh (Lift > 1) ---
    print(f"--- 5. Lọc các Luật Mạnh (Lift > {min_lift}) ---")
    
    strong_rules = rules[rules['lift'] > min_lift]
    
    # Sắp xếp theo Lift và Confidence cao nhất để xem luật tốt nhất
    strong_rules = strong_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])

    if strong_rules.empty:
        print(f"KHÔNG TÌM THẤY LUẬT MẠNH (Lift > {min_lift}). Thử điều chỉnh lại các ngưỡng.")
        return

    print("--- KẾT QUẢ: CÁC LUẬT MẠNH (INSIGHTS) ---")
    
    # Hiển thị các cột quan trọng
    final_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(strong_rules[final_columns])
    
    # In ra một ví dụ "Insight" như nhóm bạn mô tả
    print("\n--- Ví dụ Insight (Luật tốt nhất) ---")
    best_rule = strong_rules.iloc[0]
    antecedent = ", ".join(list(best_rule['antecedents'])) # Tiền đề, ví dụ: {F_Score=1}
    consequent = ", ".join(list(best_rule['consequents'])) # Kết quả, ví dụ: {M_Score=1}
    confidence_pct = best_rule['confidence'] * 100
    lift = best_rule['lift']

    print(f"Phát hiện thú vị: Những khách hàng có đặc điểm {{ {antecedent} }}")
    print(f"  ... có xu hướng {{ {consequent} }}")
    print(f"  ... với độ tin cậy {confidence_pct:.2f}% và (Lift = {lift:.2f}).")
    print("  (Vì Lift > 1, đây là một mối liên hệ mạnh, không phải ngẫu nhiên)")


# -----------------------------------------------------------------
# --- BẮT ĐẦU CHƯƠNG TRÌNH ---
# -----------------------------------------------------------------

# --- Bước 1: Chuẩn bị Dữ liệu ---

# === ĐỂ DÙNG FILE CỦA BẠN ===
# 1. Bỏ comment (dấu #) ở dòng dưới
# 2. Thay 'your_rfm_data.csv' bằng tên file của bạn
# 3. Đảm bảo file CSV có các cột tên là 'Recency', 'Frequency', 'Monetary'
#
print("Đang tải dữ liệu từ file CSV...")
try:
    main_rfm_df = pd.read_csv('rfm.csv')
    main_rfm_df = main_rfm_df.dropna(subset=['Recency', 'Frequency', 'Monetary'])
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'rfm.csv'. Vui lòng kiểm tra lại tên file.")
    exit()
except Exception as e:
    print(f"LỖI khi đọc file: {e}")
    exit()


# --- Bước 2: Điều chỉnh Ngưỡng (Parameters) ---

# CHÚ Ý QUAN TRỌNG:
# 1. min_support: Tỉ lệ tối thiểu của một nhóm.
#    - Với dữ liệu lớn, bạn nên bắt đầu ở mức rất thấp, ví dụ: 0.01 (1%) hoặc 0.005.
#    - Vì chúng ta đang dùng 4 điểm (1, 2, 3, 4), các nhóm sẽ nhỏ hơn -> min_support CÓ THỂ cần nhỏ hơn trước.
MIN_SUPPORT = 0.05 # Thử bắt đầu ở 5%

# 2. min_confidence: Độ tin cậy của luật (A -> B).
#    - 0.5 (50%) là một điểm khởi đầu tốt.
MIN_CONFIDENCE = 0.5

# 3. min_lift: Ngưỡng lọc "insight" của bạn.
#    - Luôn giữ > 1.
MIN_LIFT = 1.0


# --- Bước 3: Chạy phân tích ---
if main_rfm_df.empty:
    print("DataFrame rỗng. Vui lòng kiểm tra lại nguồn dữ liệu.")
else:
    # Sao chép DataFrame để tránh SettingWithCopyWarning
    rfm_copy = main_rfm_df.copy()
    run_association_rules_on_rfm(rfm_copy, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT)