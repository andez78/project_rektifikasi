# src/app.py

import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="KRIS AI", layout="centered", initial_sidebar_state="collapsed")

# --- 2. KONFIGURASI PATH DAN NAMA KOLOM ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
ASSETS_DIR = os.path.join(BASE_DIR, '..', 'assets')

NAMA_FILE_PROCESSED_CSV = 'processed_data_v3.csv' 
PATH_PROCESSED_FILE = os.path.join(DATA_DIR, 'processed', NAMA_FILE_PROCESSED_CSV)
PATH_TFIDF_VECTORIZER = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')

# Nama kolom
COLUMN_AIRCRAFT_TYPE = 'aircraft_type'
COLUMN_FINDING_DESC_PROC = 'finding_description'

# --- 3. FUNGSI-FUNGSI UTAMA ---
@st.cache_resource
def load_vectorizer(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

@st.cache_data
def load_processed_data(path):
    try:
        df = pd.read_csv(path)
        list_columns = ['rectification_steps', 'man_hours_per_step', 'work_centres_per_step', 'plants_per_step']
        for col in list_columns:
            if col in df.columns and not df.empty and isinstance(df[col].iloc[0], str):
                try:
                    df[col] = df[col].apply(ast.literal_eval)
                except:
                    pass
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# Fungsi get_recommendations_app tidak berubah
def get_recommendations_app(new_finding_text, aircraft_type, top_n=3):
    filtered_df = df_for_tfidf[df_for_tfidf[COLUMN_AIRCRAFT_TYPE].str.lower() == aircraft_type.lower()].reset_index(drop=True)
    if filtered_df.empty: return "NoData"
    historical_tfidf_features = tfidf_vectorizer.transform(filtered_df[COLUMN_FINDING_DESC_PROC])
    new_tfidf_features = tfidf_vectorizer.transform([new_finding_text.lower()])
    similarities = cosine_similarity(new_tfidf_features, historical_tfidf_features)
    most_similar_indices = similarities[0].argsort()[-top_n:][::-1]
    recommendations_output = []
    for i, index in enumerate(most_similar_indices):
        similarity_score = similarities[0][index]
        if similarity_score > 0.01:
            rec_detail = filtered_df.iloc[index]
            recommendations_output.append({k: rec_detail.get(k) for k in rec_detail.index})
            recommendations_output[-1]['rank'] = i + 1
            recommendations_output[-1]['similarity_score'] = similarity_score
    return recommendations_output

# --- 4. PEMUATAN DATA DAN INISIALISASI ---
tfidf_vectorizer = load_vectorizer(PATH_TFIDF_VECTORIZER)
processed_df_loaded = load_processed_data(PATH_PROCESSED_FILE)

df_for_tfidf = pd.DataFrame()
if not processed_df_loaded.empty:
    df_for_tfidf = processed_df_loaded.copy()
    df_for_tfidf.dropna(subset=[COLUMN_FINDING_DESC_PROC, COLUMN_AIRCRAFT_TYPE], inplace=True)
    df_for_tfidf[COLUMN_FINDING_DESC_PROC] = df_for_tfidf[COLUMN_FINDING_DESC_PROC].astype(str)
    df_for_tfidf[COLUMN_AIRCRAFT_TYPE] = df_for_tfidf[COLUMN_AIRCRAFT_TYPE].astype(str)
    df_for_tfidf.reset_index(drop=True, inplace=True)

if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# --- 5. TAMPILAN UI APLIKASI ---
if tfidf_vectorizer is None or df_for_tfidf.empty:
    st.error("❌ Sistem tidak siap. Gagal memuat model atau data historis.")
else:
    if st.session_state.get('search_triggered', False):
        # --- HALAMAN HASIL REKOMENDASI ---
        if st.button("⬅️ Lakukan Pencarian Baru"):
            st.session_state.search_triggered = False
            # PERUBAHAN 1: Menggunakan st.rerun()
            st.rerun()
        
        st.caption(f"Hasil rekomendasi untuk '{st.session_state.user_input}' pada pesawat {st.session_state.ac_type}")
        st.markdown("---")
        
        recommendations = st.session_state.get('recommendations', [])
        
        if recommendations == "NoData":
            st.info(f"ℹ️ Belum ada data historis untuk tipe pesawat {st.session_state.ac_type}.")
        elif not recommendations:
            st.warning("ℹ️ Tidak ditemukan rekomendasi yang cukup mirip.")
        else:
            for rec in recommendations:
                with st.container():
                    st.markdown(f"##### Rekomendasi #{rec.get('rank')} (Kemiripan: {rec.get('similarity_score'):.2%})")
                    st.markdown(f"**Finding Historis:** `{rec.get('finding_description')}`")
                    breadcrumb = (f"✈️ **Tipe Pesawat:** {rec.get('aircraft_type','N/A').upper()} | "
                                f"📋 **No. Order:** {rec.get('order_info', 'N/A')} ")
                    st.markdown(breadcrumb)
                    with st.expander("Lihat Detail Rekomendasi Lengkap"):
                        st.markdown("**🛠️ Saran Langkah Rektifikasi:**")
                        table_data = []
                        rect_steps = rec.get('rectification_steps', [])
                        if isinstance(rect_steps, list) and len(rect_steps) > 0:
                            man_hours_list = rec.get('man_hours_per_step', [])
                            work_centres_list = rec.get('work_centres_per_step', [])
                            plants_list = rec.get('plants_per_step', [])
                            min_len = min(len(rect_steps), len(man_hours_list), len(work_centres_list), len(plants_list))
                            for i in range(min_len):
                                table_data.append({
                                    "Ops": i + 1, "Langkah Rektifikasi": rect_steps[i],
                                    "Work Centre": work_centres_list[i], "Plant": plants_list[i], "Manhours": man_hours_list[i]
                                })
                            df_steps = pd.DataFrame(table_data)
                            st.dataframe(df_steps, hide_index=True, use_container_width=True)
                        else:
                            st.markdown("_(Tidak ada detail langkah rektifikasi)_")
                        
                        st.markdown("**🔩 Saran Material:**")
                        materials_text = rec.get('materials_info', 'N/A')
                        if isinstance(materials_text, str) and materials_text.lower() not in ['nan', 'n/a', '']:
                             split_materials = re.split(r'\s*[,;\n]\s*', materials_text.strip())
                             unique_materials = list(dict.fromkeys([m.strip() for m in split_materials if m.strip()]))
                             if unique_materials:
                                 for m_item in unique_materials:
                                     st.markdown(f"- {m_item}")
                             else:
                                 st.markdown("_(Tidak ada informasi material)_")
                        else:
                             st.markdown("_(Tidak ada informasi material)_")

    else:
        # --- HALAMAN PENCARIAN (TAMPILAN AWAL) ---
        col1, col2 = st.columns([2, 4], gap="medium", vertical_alignment="center")
        with col1:
            LOGO_PATH = os.path.join(ASSETS_DIR, 'logo_perusahaan.png')
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=500)
            else:
                st.caption("logo.png not found")
        with col2:
            st.markdown("<h1 style='color: #4A90E2; text-align: center; font-size: 4.5em; margin-bottom: 0px;'>KRIS AI</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Knowledge-based Rectification Insight System</p>", unsafe_allow_html=True)
        st.write("")
        
        aircraft_types = sorted(df_for_tfidf[COLUMN_AIRCRAFT_TYPE].str.upper().unique().tolist())
        
        with st.form(key='search_form'):
            ac_type_input = st.selectbox(
                "1. Pilih Tipe Pesawat", 
                aircraft_types, 
                index=None,
                placeholder="Pilih tipe pesawat..."
            )
            user_input_area = st.text_area(
                "2. Masukkan Deskripsi Finding", 
                height=150, 
                placeholder="Contoh: engine oil leak from pipe A section 2..."
            )
            submit_button = st.form_submit_button(label="🔍 Dapatkan Rekomendasi", type="primary", use_container_width=True)

        if submit_button:
            if user_input_area and ac_type_input:
                with st.spinner("⏳ Mencari rekomendasi..."):
                    recommendations = get_recommendations_app(user_input_area, ac_type_input)
                
                st.session_state.recommendations = recommendations
                st.session_state.user_input = user_input_area
                st.session_state.ac_type = ac_type_input
                st.session_state.search_triggered = True
                # PERUBAHAN 2: Menggunakan st.rerun()
                st.rerun()
            else:
                st.warning("⚠️ Silakan pilih Tipe Pesawat dan masukkan deskripsi finding terlebih dahulu.")