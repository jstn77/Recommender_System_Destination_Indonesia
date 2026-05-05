import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Rekomendasi Wisata Indonesia", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/tourism_with_id.csv')
    # Cleaning data dasar
    cols_to_drop = ['Time_Minutes', 'Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df

df = load_data()

# --- PREPROCESSING TEKS ---
def text_cleaning(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def get_similarity_matrix(data):
    # Buat tags untuk Content Based Filtering
    data['tags'] = data['Description'].apply(text_cleaning) + " " + \
                   (data['Category'].apply(text_cleaning) + " ") * 2 + \
                   (data['City'].apply(text_cleaning) + " ") * 2
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = get_similarity_matrix(df)

# --- FUNGSI REKOMENDASI ---
def get_recommendations(keyword, k=5):
    try:
        idx = df[df['Place_Name'].str.lower() == keyword.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:k+1]
        place_indices = [i[0] for i in sim_scores]
        return df.iloc[place_indices]
    except:
        return None

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman:", ["Home & Recommendation", "Exploratory Data Analysis (EDA)"])

# --- HALAMAN 1: HOME & RECOMMENDATION ---
if page == "Home & Recommendation":
    st.title("🏯 Sistem Rekomendasi Destinasi Wisata Indonesia")
    st.markdown("Cari destinasi serupa berdasarkan tempat yang Anda sukai menggunakan *Content-Based Filtering*.")
    
    # Input User
    selected_place = st.selectbox("Pilih atau Ketik Nama Tempat Wisata:", df['Place_Name'].values)
    top_k = st.slider("Jumlah Rekomendasi:", 1, 10, 5)
    
    if st.button("Berikan Rekomendasi"):
        results = get_recommendations(selected_place, k=top_k)
        
        if results is not None:
            st.success(f"Berikut adalah destinasi yang mirip dengan **{selected_place}**:")
            for i, row in results.iterrows():
                with st.expander(f"{row['Place_Name']} ({row['City']}) - ⭐ {row['Rating']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Kategori:** {row['Category']}")
                        st.write(f"**Harga Tiket:** Rp{row['Price']:,}")
                    with col2:
                        st.write(f"**Deskripsi:**")
                        st.write(row['Description'])
        else:
            st.error("Tempat tidak ditemukan dalam database.")

# --- HALAMAN 2: EDA ---
elif page == "Exploratory Data Analysis (EDA)":
    st.title("📊 Analisis Data Pariwisata")
    
    # Statistik Singkat
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Destinasi", len(df))
    col2.metric("Jumlah Kota", df['City'].nunique())
    col3.metric("Kategori Wisata", df['Category'].nunique())
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Kategori", "Distribusi Kota", "Rating vs Harga"])
    
    with tab1:
        st.subheader("Kategori Tempat Wisata Terbanyak")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index, palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Distribusi Wisata per Kota")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='City', data=df, order=df['City'].value_counts().index, palette='magma', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Hubungan Rating dan Harga Tiket")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(x='Rating', y='Price', hue='Category', data=df, ax=ax)
        st.pyplot(fig)
        
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Dibuat untuk Sistem Rekomendasi Wisata menggunakan Streamlit.")