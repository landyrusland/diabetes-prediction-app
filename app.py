import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Judul Aplikasi
st.title("Aplikasi Prediksi Diabetes")
st.write("Dibuat berdasarkan analisis Random Forest Classifier")

# --- 1. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_train_model():
    # Menggunakan Direct Link Google Drive
    file_id = '1y0gsrGyNvXf3RBSDZH98_cWYoBoMffgA'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error("Gagal memuat data. Pastikan akses file di Google Drive adalah 'Anyone with the link'.")
        return None, None, None

    # Preprocessing
    df_copy = df.copy(deep=True)
    cols_to_clean = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df_copy[cols_to_clean] = df_copy[cols_to_clean].replace(0, np.nan)

    # Mengisi NaN dengan Median
    for col in cols_to_clean:
        df_copy[col].fillna(df_copy[col].median(), inplace=True)

    # Split Data
    X = df_copy.drop('Outcome', axis=1)
    y = df_copy['Outcome']
    
    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
    
    rf = RandomForestClassifier(random_state=8)
    rf.fit(X_train, y_train)
    
    return rf, df_copy, X_test.columns

model, df_clean, feature_names = load_and_train_model()

if model is not None:
    # --- 2. SIDEBAR INPUT USER ---
    st.sidebar.header("Masukkan Data Pasien")

    def user_input_features():
        Pregnancies = st.sidebar.number_input('Pregnancies (Kehamilan)', min_value=0, max_value=20, value=1)
        Glucose = st.sidebar.slider('Glucose (Glukosa)', 0, 200, 120)
        BloodPressure = st.sidebar.slider('Blood Pressure (Tekanan Darah)', 0, 130, 70)
        SkinThickness = st.sidebar.slider('Skin Thickness (Ketebalan Kulit)', 0, 100, 20)
        Insulin = st.sidebar.slider('Insulin', 0, 900, 30)
        BMI = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)
        DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
        Age = st.sidebar.slider('Age (Umur)', 1, 100, 25)

        data = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Tampilkan Input User
    st.subheader('Data Pasien:')
    st.write(input_df)

    # --- 3. PREDIKSI ---
    if st.button('Prediksi'):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.error(f'Positif Diabetes (Probabilitas: {prediction_proba[0][1]*100:.2f}%)')
        else:
            st.success(f'Negatif Diabetes (Probabilitas: {prediction_proba[0][0]*100:.2f}%)')