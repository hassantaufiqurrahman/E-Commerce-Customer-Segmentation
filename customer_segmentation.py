import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

print("🔄 Menjalankan backend pipeline data & business profiling...")

# Load Dataset
file_path = 'data/ecommerce_customer_behavior_dataset_v2.csv'
if not os.path.exists(file_path):
    file_path = '../data/ecommerce_customer_behavior_dataset_v2.csv'

df = pd.read_csv(file_path)

# Membersihkan baris dengan Gender 'Other'
df = df[df['Gender'].isin(['Female', 'Male'])].copy()

# Konversi Tanggal & Tentukan Snapshot Date
df['Date'] = pd.to_datetime(df['Date'])
snapshot_date = df['Date'].max() + pd.Timedelta(days=1)

# Agregasi RFM Mentah per Customer_ID
rfm_df = df.groupby('Customer_ID').agg(
    Recency=('Date', lambda x: (snapshot_date - x.max()).days),
    Frequency=('Order_ID', 'nunique'),
    Monetary=('Total_Amount', 'sum')
).reset_index()

# Filter IQR khusus Monetary
def filter_rfm_iqr(df_in, features):
    df_clean = df_in.copy()
    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

df_cluster_clean = filter_rfm_iqr(rfm_df, ['Monetary'])

# Scaling Fitur RFM
rfm_features = ['Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster_clean[rfm_features])

# Pembuatan Klaster K-Means
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cluster_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Agregasi Perilaku Tambahan
customer_behavior_agg = df.groupby('Customer_ID').agg(
    Rata2_Unit_Price=('Unit_Price', 'mean'),
    Rata2_Discount=('Discount_Amount', 'mean')
).reset_index()

# Merge ke dataframe klaster bersih
df_profile_full = df_cluster_clean.merge(customer_behavior_agg, on='Customer_ID', how='inner')

# Membentuk Tabel Business Profiling Utama
business_profiling = df_profile_full.groupby('Cluster').agg(
    Jumlah_Customer=('Customer_ID', 'count'),
    Recency_Rata2=('Recency', 'mean'),
    Frequency_Rata2=('Frequency', 'mean'),
    Monetary_Rata2=('Monetary', 'mean'),
    Rata2_Unit_Price=('Rata2_Unit_Price', 'mean'),
    Rata2_Discount=('Rata2_Discount', 'mean')
).reset_index()

# Hitung Persentase Jumlah Customer
business_profiling['Persentase_Customer (%)'] = (
    business_profiling['Jumlah_Customer'] / len(df_cluster_clean) * 100
).round(2)

# Pemetaan Nama Klaster Deskriptif
cluster_name_mapping = {
    0: "Regular Customers",
    1: "Churn Risk Customers",
    2: "VIP Customers"
}

business_profiling['Cluster_Name'] = business_profiling['Cluster'].map(cluster_name_mapping)

# Gabungkan Label Klaster ke Transaksi Asli
df_transaksi_cluster = df.merge(df_cluster_clean[['Customer_ID', 'Cluster']], on='Customer_ID', how='inner')
df_transaksi_cluster['Cluster_Name'] = df_transaksi_cluster['Cluster'].map(cluster_name_mapping)

# Save File Artefak ke Folder result/
os.makedirs('result', exist_ok=True)
joblib.dump(scaler, 'result/scaler.pkl')
joblib.dump(kmeans, 'result/kmeans_model.pkl')
business_profiling.to_csv('result/cluster_business_profiling.csv', index=False)
df_transaksi_cluster.to_csv('result/df_transaksi_cluster.csv', index=False)

print("✅ Semua artefak backend (termasuk penamaan klaster bisnis) berhasil disimpan di folder 'result/'!")