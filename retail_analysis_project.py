"""
Retail Analytics Project: Customer Segmentation (RFM) & Market Basket Analysis
Author: [Your Name]
Description: This script performs deep dive analysis on retail data to extract 
business insights using RFM modeling and Association Rules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# --- 1. Configuration & Styling ---
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- 2. Data Cleaning Function ---
def clean_data(df):
    """Clean dataset: handle missing values and negative transactions."""
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Total_Price'] = df['Quantity'] * df['Price']
    return df

# --- 3. RFM Analysis & Scoring ---
def perform_rfm_analysis(df):
    """Calculate R, F, M scores and segment customers."""
    today_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (today_date - date.max()).days,
        'Invoice': lambda num: num.nunique(),
        'Total_Price': lambda price: price.sum()
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Scoring (1-5) using qcut
    rfm["R_Score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["M_Score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    # Create RFM Score string
    rfm["RFM_Score"] = (rfm['R_Score'].astype(str) + 
                        rfm['F_Score'].astype(str) + 
                        rfm['M_Score'].astype(str))
    
    # Segment Mapping (As shown in your results)
    seg_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Cant Lose Them',
        r'3[1-2]': 'About to Sleep',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising',
        r'51': 'New Customers',
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    # Segmenting based on R and F scores
    rfm['Segment'] = (rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)).replace(seg_map, regex=True)
    return rfm

# --- 4. Market Basket Analysis ---
def market_basket_analysis(df):
    """Extract product associations using Apriori algorithm."""
    basket = (df.groupby(['Invoice', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('Invoice'))
    
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    return rules.sort_values("lift", ascending=False)

# --- 5. Visualization Suite ---
def plot_results(rfm, rules):
    # Chart 1: Segment Distribution (Bar Chart)
    plt.figure(figsize=(12, 6))
    segment_counts = rfm['Segment'].value_counts()
    sns.barplot(x=segment_counts.values, y=segment_counts.index, palette='viridis')
    plt.title('Distribution of Customers Across RFM Segments')
    plt.show()

    # Chart 2: RFM Score Histograms (As shown in your images)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['skyblue', 'salmon', 'lightgreen']
    for i, (col, color) in enumerate(zip(['R_Score', 'F_Score', 'M_Score'], colors)):
        sns.histplot(rfm[col].astype(int), kde=True, ax=axes[i], color=color)
        axes[i].set_title(f'{col} Distribution')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
# Note: In a real environment, you would load your data here:
# df = pd.read_csv('your_data.csv')
# df_cleaned = clean_data(df)
# rfm_results = perform_rfm_analysis(df_cleaned)
# association_rules_df = market_basket_analysis(df_cleaned)
# plot_results(rfm_results, association_rules_df)
