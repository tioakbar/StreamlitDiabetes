import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================================
# Load model
# ================================
with open("XGBM_model.pkl", "rb") as file:
    model = pickle.load(file)

# ================================
# Konfigurasi halaman
# ================================
st.set_page_config(page_title="Prediksi Diabetes", page_icon="💉", layout="centered")

st.title("💉 Aplikasi Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi kemungkinan terkena diabetes menggunakan model XGBoost.")

# ================================
# Input user
# ================================
st.header("Masukkan Data Pasien")

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

# ================================
# Prediksi
# ================================
if st.button("Prediksi"):
    # Membuat DataFrame input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_df = pd.DataFrame(input_data, columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])
    
    # Prediksi
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    # ================================
    # Hasil Prediksi
    # ================================
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error(f"⚠️ Pasien **berpotensi diabetes** (Probabilitas: {proba:.2%})")
    else:
        st.success(f"✅ Pasien **tidak berpotensi diabetes** (Probabilitas: {proba:.2%})")

# ================================
# Footer
# ================================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan XGBoost")
