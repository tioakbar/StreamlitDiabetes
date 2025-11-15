import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# ================================
# Load model & data
# ================================
with open("XGBM_model.pkl", "rb") as file:
    model = pickle.load(file)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("diabetes_data.csv")
    except FileNotFoundError:
        df = None
    return df

data = load_data()

# ================================
# Konfigurasi halaman
# ================================
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="💉",
    layout="wide"
)

# ================================
# Sidebar Navigasi
# ================================
st.sidebar.title("💊 Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dashboard Data", "Prediksi Pasien"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dibuat dengan ❤️ menggunakan Streamlit dan XGBoost")

# ================================
# HALAMAN 1: DASHBOARD DATA
# ================================
if page == "Dashboard Data":
    st.title("📊 Dashboard Data Diabetes")

    if data is None:
        st.error(
            "File **diabetes_data.csv** tidak ditemukan. "
            "Pastikan file tersebut berada di folder yang sama dengan aplikasi."
        )
    else:
        st.markdown(
            "Dashboard ini menampilkan gambaran umum dataset diabetes yang digunakan "
            "untuk melatih model prediksi."
        )

        # Bersihkan nama kolom (jaga-jaga)
        data.columns = [c.strip() for c in data.columns]

        # Coba deteksi nama kolom standar
        col_glucose = "Glucose" if "Glucose" in data.columns else None
        col_bmi = "BMI" if "BMI" in data.columns else None
        col_age = "Age" if "Age" in data.columns else None
        col_outcome = "Outcome" if "Outcome" in data.columns else None

        # ---------- KPI / METRIC ----------
        st.markdown("### 🔍 Ringkasan Umum")

        total_data = len(data)
        mean_glucose = data[col_glucose].mean() if col_glucose else 0
        mean_bmi = data[col_bmi].mean() if col_bmi else 0

        if col_outcome:
            diabetes_rate = data[col_outcome].mean() * 100
        else:
            diabetes_rate = 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Jumlah Data", f"{total_data}")
        m2.metric("Rata-rata Glukosa", f"{mean_glucose:.1f}" if mean_glucose else "-")
        m3.metric("Persentase Diabetes", f"{diabetes_rate:.1f}%" if diabetes_rate else "-")

        st.markdown("---")

        # ---------- Visualisasi Distribusi ----------
        st.markdown("### 📈 Distribusi Fitur Utama")

        col_a, col_b = st.columns(2)

        with col_a:
            if col_glucose:
                st.subheader("Distribusi Glukosa")
                st.bar_chart(data[col_glucose].value_counts().sort_index())
            else:
                st.info("Kolom **Glucose** tidak ditemukan di dataset.")

        with col_b:
            if col_bmi:
                st.subheader("Distribusi BMI")
                bmi_sorted = data[col_bmi].dropna().sort_values().reset_index(drop=True)
                st.line_chart(bmi_sorted)
            else:
                st.info("Kolom **BMI** tidak ditemukan di dataset.")

        st.markdown("---")

        # ---------- Hubungan Dua Variabel ----------
        st.markdown("### 🔗 Hubungan Usia dan BMI")

        if col_age and col_bmi:
            scatter_df = data[[col_age, col_bmi]].dropna()
            st.caption("Setiap titik merepresentasikan satu pasien (Age vs. BMI).")
            st.scatter_chart(scatter_df, x=col_age, y=col_bmi)
        else:
            st.info("Kolom **Age** atau **BMI** tidak tersedia untuk scatter plot.")

        if col_outcome:
            st.markdown("---")
            st.markdown("### 🩺 Perbandingan Jumlah Pasien Diabetes vs Tidak Diabetes")
            outcome_counts = data[col_outcome].value_counts().rename(index={0: "Tidak Diabetes", 1: "Diabetes"})
            st.bar_chart(outcome_counts)

        st.markdown("---")
        st.caption(
            "Dashboard ini dapat dikembangkan lebih lanjut dengan menambah filter, "
            "segmentasi kelompok usia, maupun analisis lanjutan lainnya."
        )


# ================================
# HALAMAN 2: PREDIKSI PASIEN
# ================================
elif page == "Prediksi Pasien":
    st.title("💉 Aplikasi Prediksi Diabetes")
    st.write(
        "Masukkan data pasien untuk memprediksi kemungkinan terkena diabetes "
        "menggunakan model **XGBoost** yang telah dilatih."
    )

    st.markdown("### 🧾 Formulir Data Pasien")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, step=1)
        glucose = st.number_input("Glukosa (mg/dL)", min_value=0)
        blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0)
        skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0)

    with col2:
        insulin = st.number_input("Insulin (µU/mL)", min_value=0)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, format="%.2f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        age = st.number_input("Usia (tahun)", min_value=0, step=1)

    col_pred, col_info = st.columns([2, 1])

    with col_pred:
        if st.button("🔍 Prediksi"):
            progress_text = "Sedang memproses prediksi..."
            progress_bar = st.progress(0, text=progress_text)

            for i in range(0, 101, 10):
                time.sleep(0.05)
                progress_bar.progress(i, text=progress_text)

            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])
            input_df = pd.DataFrame(input_data, columns=[
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ])

            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            st.markdown("### 📊 Hasil Prediksi")

            if prediction == 1:
                st.error(
                    f"⚠️ Pasien **berpotensi diabetes** (Probabilitas: {proba:.2%})"
                )
            else:
                st.success(
                    f"✅ Pasien **tidak berpotensi diabetes** (Probabilitas: {proba:.2%})"
                )
                st.balloons()

    with col_info:
        st.markdown("### ℹ️ Tips Penggunaan")
        st.write(
            "- Lengkapi semua input.\n"
            "- Hasil prediksi bukan diagnosis medis.\n"
            "- Gunakan sebagai alat bantu edukasi."
        )

    st.markdown("---")
    st.caption("Catatan: Model memiliki batasan dan potensi error.")
