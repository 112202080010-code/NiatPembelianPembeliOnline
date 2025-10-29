import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="🛒 Online Shopper Purchase Prediction", layout="centered")

# --- Header ---
st.title("🛒 Prediksi Perilaku Pembelian Online")
st.markdown("""
Aplikasi ini memprediksi **kemungkinan pengunjung melakukan pembelian (Revenue)**  
berdasarkan perilaku mereka di website toko online.

Model menggunakan beberapa fitur utama seperti:
- 📄 Aktivitas Halaman (Administrative, Informational, ProductRelated)
- 💻 Durasi dan Tingkat Keluar (BounceRates, ExitRates, PageValues)
- 📅 Bulan dan Hari Khusus (Month, SpecialDay)
- 👤 Jenis Pengunjung dan Hari Weekend
""")

st.divider()

# --- Upload Dataset (Opsional) ---
st.subheader("📂 Upload Dataset (Opsional)")
uploaded_file = st.file_uploader("Unggah file dataset (.csv) untuk melihat data:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📋 **5 Data Teratas dari Dataset Anda:**")
    st.dataframe(df.head())

    st.subheader("📊 Statistik Deskriptif")
    st.write(df.describe())

    if "Revenue" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        df["Revenue"].astype(int).value_counts().plot(kind='bar', color=["salmon", "skyblue"], ax=ax)
        ax.set_xticklabels(["Tidak Beli (0)", "Beli (1)"], rotation=0)
        ax.set_title("Distribusi Target (Revenue)")
        st.pyplot(fig)
else:
    st.info("Belum ada dataset diunggah. Anda tetap bisa melakukan prediksi manual di bawah ini 👇")

st.divider()

# --- Input Data untuk Prediksi ---
st.header("📥 Masukkan Data Pengunjung Website")

col1, col2, col3 = st.columns(3)
with col1:
    administrative = st.number_input("Administrative:", min_value=0, max_value=30, value=2)
    informational = st.number_input("Informational:", min_value=0, max_value=10, value=1)
    product_related = st.number_input("ProductRelated:", min_value=0, max_value=500, value=50)
with col2:
    bounce_rates = st.number_input("BounceRates:", min_value=0.0, max_value=1.0, value=0.02)
    exit_rates = st.number_input("ExitRates:", min_value=0.0, max_value=1.0, value=0.05)
    page_values = st.number_input("PageValues:", min_value=0.0, max_value=100.0, value=20.0)
with col3:
    special_day = st.number_input("SpecialDay (0–1):", min_value=0.0, max_value=1.0, value=0.0)
    month = st.selectbox("Month:", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    visitor_type = st.selectbox("VisitorType:", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.selectbox("Weekend:", ["TRUE", "FALSE"])

# --- Prediksi ---
if st.button("🔮 Prediksi Kemungkinan Pembelian"):
    try:
        # Load model, scaler, dan encoder
        model = joblib.load("model_rf_online_shoppers.joblib")
        scaler = joblib.load("scaler_online_shoppers.joblib")
        encoders = joblib.load("encoders_online_shoppers.joblib")

        # Buat DataFrame input
        new_data = pd.DataFrame({
            'Administrative': [administrative],
            'Administrative_Duration': [0],  # opsional jika tidak diminta
            'Informational': [informational],
            'Informational_Duration': [0],
            'ProductRelated': [product_related],
            'ProductRelated_Duration': [0],
            'BounceRates': [bounce_rates],
            'ExitRates': [exit_rates],
            'PageValues': [page_values],
            'SpecialDay': [special_day],
            'Month': [month],
            'VisitorType': [visitor_type],
            'Weekend': [weekend]
        })

        # Encode kategori
        for col in ['Month', 'VisitorType', 'Weekend']:
            if col in encoders:
                new_data[col] = encoders[col].transform(new_data[col])

        # Standarisasi fitur numerik
        num_features = [
            'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
        ]
        scaled_data = new_data.copy()
        scaled_data[num_features] = scaler.transform(new_data[num_features])

        # Prediksi
        pred = model.predict(scaled_data)[0]
        prob = np.clip(pred, 0, 1)  # batasi antara 0–1

        # --- Hasil Prediksi ---
        st.success(f"🧠 **Kemungkinan Pengunjung Melakukan Pembelian: {prob*100:.2f}%**")

        # --- Visualisasi ---
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Kemungkinan Pembelian"], [prob*100], color="green")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Persentase (%)")
        ax.set_title("Prediksi Peluang Pembelian")
        st.pyplot(fig)

        # --- Interpretasi ---
        if prob > 0.6:
            st.success("🎯 Prediksi: **Kemungkinan besar pengunjung akan melakukan pembelian.**")
        elif prob > 0.3:
            st.warning("⚖️ Prediksi: **Kemungkinan sedang untuk pembelian.**")
        else:
            st.info("🕓 Prediksi: **Kemungkinan kecil pengunjung akan membeli.**")

    except FileNotFoundError:
        st.error("⚠️ File model/scaler/encoder tidak ditemukan. Pastikan semua file joblib tersedia di folder proyek Anda.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Dibuat oleh: **Suwannur32** | Project: Online Shopper Purchase Prediction 🛒 | Powered by Streamlit & scikit-learn")
