import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import datetime
import os

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Customer Cluster Prediction",
    page_icon="🎯",
    layout="wide"
)

# Load Aset dari Hasil Backend
@st.cache_resource
def load_app_assets():
    scaler = joblib.load('result/scaler.pkl')
    kmeans_model = joblib.load('result/kmeans_model.pkl')
    business_profiling = pd.read_csv('result/cluster_business_profiling.csv')
    df_transaksi_cluster = pd.read_csv('result/df_transaksi_cluster.csv')
    return scaler, kmeans_model, business_profiling, df_transaksi_cluster

# Validasi Kelengkapan File Artefak Backend
required_files = [
    'result/scaler.pkl', 
    'result/kmeans_model.pkl', 
    'result/cluster_business_profiling.csv', 
    'result/df_transaksi_cluster.csv'
]
for f in required_files:
    if not os.path.exists(f):
        st.error(f"⚠️ File `{f}` not found! Please run the `customer_segmentation.py` script first.")
        st.stop()

scaler, kmeans_model, business_profiling, df_transaksi_cluster = load_app_assets()

# Menentukan "snapshot_date" Berdasarkan Backend
if 'Date' in df_transaksi_cluster.columns:
    df_transaksi_cluster['Date'] = pd.to_datetime(df_transaksi_cluster['Date'])
    snapshot_date = df_transaksi_cluster['Date'].max().date() + datetime.timedelta(days=1)
else:
    snapshot_date = datetime.date(2024, 3, 26)

# Inisialisasi Session State untuk Mengontrol Status Form & Hasil Prediksi
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

st.title("🎯 Customer Cluster Prediction")
st.write(
    "An interactive application for predicting a customer's cluster based on recency, frequency, and monetary (RFM) metrics, while also displaying the profiles and characteristics of the predicted clusters."
    " Created by Hassan Taufiqurrahman."
)

st.info(f"📅 **Reference Date:** `{snapshot_date}`")

# Form Prediksi Klaster Pelanggan
st.markdown("---")
st.subheader("📝 Customer Data Input Form")

with st.form("prediksi_form"):
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        customer_name = st.text_input(
            "Customer Label / Name:",
            value="",
            placeholder="e.g. Customer A / John",
            help="Free-text labels for naming in prediction results."
        )
        
    with col_f2:
        last_transaction_date = st.date_input(
            "Customer's Last Date:",
            value=snapshot_date - datetime.timedelta(days=15),
            max_value=snapshot_date - datetime.timedelta(days=1),
            help="The customer's last transaction date."
        )
        
    with col_f3:
        frequency_input = st.number_input(
            "Order Quantity (Frequency):",
            value=None,
            min_value=1,
            max_value=10,
            step=1,
            placeholder="Insert frequency (e.g. 3)",
            help="Total order frequency."
        )
        
    with col_f4:
        monetary_input = st.number_input(
            "Total Amount (Monetary in TRY):",
            value=None,
            min_value=0.0,
            max_value=50628.15,
            step=10.0,
            placeholder="Insert amount (e.g. 500)",
            help="Total accumulated spend in TRY."
        )
        
    submit_btn = st.form_submit_button("🔍 Check Customer's Clusters", use_container_width=True)

if submit_btn:
    st.session_state.form_submitted = True
    st.session_state.frequency_input = frequency_input
    st.session_state.monetary_input = monetary_input
    st.session_state.customer_name = customer_name
    st.session_state.last_transaction_date = last_transaction_date

# Hasil Prediksi & Profil Khusus Klaster Terkait
if st.session_state.form_submitted:
    # Validasi input angka yang wajib diisi
    if frequency_input is None or monetary_input is None:
        st.warning("⚠️ Please fill in all required transaction fields (Frequency & Total Amount) before proceeding.")
    else:
        display_name = customer_name.strip() if customer_name.strip() else "Customer"
        
        # Hitung Recency
        recency_calculated = (snapshot_date - last_transaction_date).days
        if recency_calculated < 0:
            recency_calculated = 0
            
        # Model hanya memproses 3 fitur RFM
        user_input_raw = [[recency_calculated, frequency_input, monetary_input]]
        user_input_scaled = scaler.transform(user_input_raw)
        predicted_cluster = kmeans_model.predict(user_input_scaled)[0]
        
        # Ambil profil dan nama klaster deskriptif berdasarkan klaster terprediksi
        pred_profile = business_profiling[business_profiling['Cluster'] == predicted_cluster].iloc[0]
        cluster_display_name = pred_profile['Cluster_Name']
        
        st.markdown("---")
        st.success(f"🎉 **Prediction Result:** **{display_name}** belongs to **{cluster_display_name}**")
        
        # Summary Metrics
        st.markdown(f"### 📊 {cluster_display_name}: Profile Summary & Benchmark Statistics")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Recency", f"{round(pred_profile['Recency_Rata2']):,.0f} Days")
        col2.metric("Average Frequency", f"{round(pred_profile['Frequency_Rata2']):,.0f} Times")
        col3.metric("Average Monetary", f"{pred_profile['Monetary_Rata2']:,.2f} TRY")
        
        st.write("") 
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Average Unit Price", f"{pred_profile['Rata2_Unit_Price']:,.2f} TRY")
        col5.metric("Average Discount Amount", f"{pred_profile['Rata2_Discount']:,.2f} TRY")
        
        # Visualizations 
        st.markdown("---")
        st.subheader(f"📈 {cluster_display_name}: Top 3 Product Categories")
        
        df_sub = df_transaksi_cluster[df_transaksi_cluster['Cluster'] == predicted_cluster]
        
        if 'Product_Category' in df_sub.columns and not df_sub.empty:
            top_3_prod = df_sub['Product_Category'].value_counts().head(3).reset_index()
            top_3_prod.columns = ['Product_Category', 'Transaction_Count']

            fig_bar = px.bar(
                top_3_prod,
                x='Transaction_Count',
                y='Product_Category',
                orientation='h',
                text='Transaction_Count',
                color='Product_Category',
                color_discrete_sequence=['#2E8B57', "#F2E420", '#6A5ACD'], 
                labels={'Product_Category': 'Product Category', 'Transaction_Count': 'Transaction Count'}
            )
            fig_bar.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tombol Reset / Input New Customer
        st.markdown("---")
        if st.button("🔄 Input New Customer / Reset Form", use_container_width=True):
            st.session_state.form_submitted = False
            st.rerun()