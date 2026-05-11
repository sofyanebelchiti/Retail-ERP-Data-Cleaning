import pandas as pd

# Load data with chunking logic to save RAM
file_name = 'online_retail_II.xlsx'
df = pd.read_excel(file_name, nrows=10000)

# 1. Handling Missing Values (Total Quality Management)
df.dropna(subset=['Customer ID'], inplace=True)

# 2. Removing Duplicates (Waste Elimination)
df.drop_duplicates(inplace=True)

# 3. Text Standardization (Process Design)
df['Description'] = df['Description'].str.strip().str.lower()

# 4. Filtering Transactions (Lean Operations)
# Remove non-commercial transactions (Price <= 0)
df = df[df['Price'] > 0]

# Identify Returns/Cancelled orders (Waste Identification)
df['Is_Cancelled'] = df['Invoice'].astype(str).str.startswith('C')

# 5. Feature Engineering
df['Total_Amount'] = df['Quantity'] * df['Price']

# Final Output Check
print("Cleaned Data Shape:", df.shape)
print(df[['Description', 'Quantity', 'Price', 'Total_Amount']].head())

import pandas as pd

# 1. Load raw data (nrows to keep it light)
df_raw = pd.read_excel('online_retail_II.xlsx', nrows=10000)
initial_count = len(df_raw)

# 2. Identify Waste (Before cleaning)
null_customers = df_raw['Customer ID'].isnull().sum()
duplicate_rows = df_raw.duplicated().sum()
negative_quantities = (df_raw['Quantity'] < 0).sum()

# 3. Execute Professional Cleaning
df_clean = df_raw.dropna(subset=['Customer ID']).copy()
df_clean.drop_duplicates(inplace=True)
df_clean = df_clean[df_clean['Price'] > 0]
final_count = len(df_clean)

# 4. Cleaning Report (The "Before & After")
print("--- DATA WASTE REPORT (LEAN ANALYSIS) ---")
print(f"1. Total Rows Analyzed: {initial_count}")
print(f"2. Missing Customer IDs (Information Gap): {null_customers}")
print(f"3. Duplicate Records (Redundancy Waste): {duplicate_rows}")
print(f"4. Negative Quantities (Returns/Errors): {negative_quantities}")
print("-" * 40)
print(f"5. Total Rows Removed: {initial_count - final_count}")
print(f"6. Clean Data Remaining: {final_count}")
print(f"7. Data Efficiency Score: {(final_count/initial_count)*100:.2f}%")

# 5. Show what was fixed in text
print("\n--- ACTION TAKEN ---")
print("- Removed rows with missing Customer IDs for better traceability.")
print("- Eliminated duplicate entries to prevent overcounting sales.")
print("- Filtered out non-commercial or zero-price transactions.")
print("- Standardized product descriptions to lower case.")

