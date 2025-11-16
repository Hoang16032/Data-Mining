import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

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

def run_association_rules_on_rfm(rfm_df, min_support=0.01, min_confidence=0.5, min_lift=1.0):
    quantiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile([.25, .5, .75]).to_dict()

    rfm_df['R_Score'] = rfm_df['Recency'].apply(r_score, args=(quantiles,))
    rfm_df['F_Score'] = rfm_df['Frequency'].apply(fm_score, args=('Frequency', quantiles))
    rfm_df['M_Score'] = rfm_df['Monetary'].apply(fm_score, args=('Monetary', quantiles))

    rfm_scores_str = rfm_df[['R_Score', 'F_Score', 'M_Score']].astype(str).dropna()
    basket_df = pd.get_dummies(rfm_scores_str, prefix_sep='=').astype(bool)

    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return None

    strong_rules = rules[rules['lift'] > min_lift]
    strong_rules = strong_rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])

    if strong_rules.empty:
        return None

    result = strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    print(result)
    return result

if __name__ == "__main__":
    try:
        main_rfm_df = pd.read_csv('rfm_datamart.csv')
        main_rfm_df = main_rfm_df.dropna(subset=['Recency', 'Frequency', 'Monetary'])
    except:
        exit()

    MIN_SUPPORT = 0.05
    MIN_CONFIDENCE = 0.5
    MIN_LIFT = 1.0

    if not main_rfm_df.empty:
        run_association_rules_on_rfm(
            main_rfm_df.copy(),
            MIN_SUPPORT,
            MIN_CONFIDENCE,
            MIN_LIFT
        )
