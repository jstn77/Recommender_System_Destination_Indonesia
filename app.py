import streamlit as st
import pandas as pd
import numpy as np
import difflib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="IndoExplore — Wisata Recommender",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- GLOBAL CSS ---
st.markdown("""
<style>
    :root {
        --bg: #ffffff; --fg: #09090b; --fg-2: #71717a; --muted: #a1a1aa;
        --accent: #3b82f6; --accent-light: rgba(59, 130, 246, 0.12);
        --border: #e4e4e7; --card: #f8fafc;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #09090b; --fg: #fafafa; --fg-2: #a1a1aa; --muted: #71717a;
            --accent: #3b82f6; --accent-light: rgba(59, 130, 246, 0.2);
            --border: #27272a; --card: #18181b;
        }
    }
    .metric-card { background: var(--card); border: 1px solid var(--border); padding: 1rem; border-radius: 8px; text-align: center; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: var(--fg); }
    .metric-label { font-size: 0.8rem; color: var(--fg-2); text-transform: uppercase; letter-spacing: 0.05em; }
    .page-header { margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
    .page-header-title { font-size: 2rem; font-weight: 800; color: var(--fg); margin-bottom: 0.2rem; }
    .page-header-purpose { font-size: 1rem; color: var(--fg-2); }
    .page-header-steps { font-size: 0.75rem; color: var(--accent); font-weight: 600; text-transform: uppercase; margin-top: 0.5rem; }
    .chart-title { font-size: 1.1rem; font-weight: 600; color: var(--fg); margin-bottom: 1rem; }
    .info-box { background: var(--card); border-left: 4px solid var(--accent); padding: 1rem; border-radius: 4px; margin-bottom: 1rem; color: var(--fg-2); font-size: 0.9rem; }
    .member-card { text-align: center; padding: 1rem; background: var(--card); border: 1px solid var(--border); border-radius: 8px; }
    .member-avatar { font-size: 2rem; margin-bottom: 0.5rem; }
    .member-name { font-weight: 600; font-size: 0.9rem; color: var(--fg); }
    .member-id { font-size: 0.8rem; color: var(--fg-2); }
</style>
""", unsafe_allow_html=True)

# ── PENGATURAN WARNA GRAFIK ──────────────────────────────────────────────────
try:
    _base = st.get_option("theme.base") or "dark"
except Exception:
    _base = "dark"

LIGHT = (_base == "light")

CHART_BG   = "#f4f4f5" if LIGHT else "#18181b"
AXIS_COLOR = "#e4e4e7" if LIGHT else "#27272a"
TEXT_COLOR = "#71717a"
FG_COLOR   = "#09090b" if LIGHT else "#fafafa"
ACCENT     = "#2563eb" if LIGHT else "#3b82f6"

plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 9,
    "axes.titlesize"   : 10,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "figure.dpi"       : 130,
})

def styled_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    return fig, ax

def apply_chart_style(fig, ax):
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(FG_COLOR)
    ax.title.set_fontweight("600")
    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS_COLOR)
    ax.grid(color=AXIS_COLOR, linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

def bar_colors(n, base=ACCENT, dim=None):
    dim = dim or AXIS_COLOR
    r0, g0, b0 = mcolors.to_rgb(dim)
    r1, g1, b1 = mcolors.to_rgb(base)
    return [(r0 + (r1-r0)*t, g0 + (g1-g0)*t, b0 + (b1-b0)*t) for t in np.linspace(0.35, 1.0, n)]

def add_bar_labels(ax, bars, fmt="{:.0f}", color=None):
    color = color or TEXT_COLOR
    max_v = max(b.get_width() for b in bars) if hasattr(bars[0], "get_width") else max(b.get_height() for b in bars)
    for bar in bars:
        is_h = hasattr(bar, "get_width") and bar.get_width() > 0
        if is_h:
            v = bar.get_width()
            ax.text(v + max_v * 0.015, bar.get_y() + bar.get_height() / 2, fmt.format(v), va="center", ha="left", fontsize=7.5, color=color)
        else:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + max_v * 0.01, fmt.format(v), va="bottom", ha="center", fontsize=7.5, color=color)

# ── DATA LOADERS & ML PIPELINE ──────────────────────────────────────────────────

@st.cache_data
def load_raw_data():
    # Load data mentah murni untuk ditampilkan di menu "Dataset"
    df_raw = pd.read_csv("dataset/tourism_with_id.csv")
    return df_raw

@st.cache_data
def load_processed_data():
    # BACA DATA YANG SUDAH BERSIH DARI JUPYTER NOTEBOOK
    # Semua proses Sastrawi, drop NA, dan text_cleaning sudah tidak ada di sini
    df_clean = pd.read_csv("dataset/tourism_cleaned.csv")
    
    # Pastikan kolom tags tidak ada yang bernilai NaN
    df_clean['tags'] = df_clean['tags'].fillna('') 
    return df_clean

@st.cache_resource
def build_model(data):
    # Langsung masukkan kolom 'tags' yang sudah bersih ke TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim

df_raw = load_raw_data()
df_clean = load_processed_data()
tfidf_matrix, cosine_sim = build_model(df_clean)

valid_cities = df_clean['City'].unique().tolist()
valid_categories = df_clean['Category'].unique().tolist()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="font-size:2.5rem;">🏝️</div>
        <div>
            <div style="font-size:1.5rem;font-weight:800;color:var(--fg);letter-spacing:-0.03em;">IndoExplore</div>
            <div style="font-size:0.8rem;color:var(--fg-2);">Sistem Rekomendasi Wisata</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("Menu", "🏠  Menu"),
        ("Dataset", "📂  Dataset"),
        ("EDA", "📊  EDA"),
        ("Preprocessing", "⚙️  Preprocessing"),
        ("Training", "🧠  Training / Evaluasi"),
        ("Prediction", "🔮  Rekomendasi"),
        ("About Us", "👥  About Us"),
    ]
    
    if "menu_choice" not in st.session_state:
        st.session_state.menu_choice = "Menu"

    for index, (page_name, label) in enumerate(nav_items, start=1):
        button_type = "primary" if page_name == st.session_state.menu_choice else "secondary"
        if st.button(label, key=f"sidebar_nav_{page_name}", use_container_width=True, type=button_type):
            st.session_state.menu_choice = page_name
            st.rerun()

    menu = st.session_state.menu_choice

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:0.8rem;color:var(--fg-2);">
        Group 02<br>ML Destination Recommender
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MENU (HOMEPAGE)
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Menu":
    total_records = len(df_clean)
    total_cities  = df_clean["City"].nunique()
    total_categories = df_clean["Category"].nunique()

    st.markdown(
        f"""
        <div style="
            text-align:center; padding:3rem 2rem 2.5rem 2rem; border-radius:12px;
            background:linear-gradient(135deg,rgba(59,130,246,0.18) 0%,rgba(16,185,129,0.15) 100%);
            border:1px solid rgba(59,130,246,0.25); margin-bottom:2rem;
        ">
            <div style="font-size:3.5rem;margin-bottom:0.4rem;">⛰️</div>
            <div style="font-size:2.6rem;font-weight:800;letter-spacing:-0.04em;color:var(--fg);margin-bottom:0.5rem;line-height:1.1;">
                Indo<span style="color:var(--accent)">Explore</span>
            </div>
            <div style="font-size:1rem;color:var(--fg-2);max-width:550px;margin:0 auto 1.5rem auto;line-height:1.6;">
                Jelajahi keindahan Indonesia. Sistem ini menggunakan <b>Content-Based Filtering</b> (TF-IDF & Cosine Similarity) untuk memberikan rekomendasi wisata personal terbaik untuk Anda.
            </div>
            <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;">
                <div style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.35); border-radius:999px;padding:0.4rem 1.1rem;font-size:0.82rem;color:var(--fg);font-weight:600;">
                    {total_records:,} Destinasi
                </div>
                <div style="background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.35); border-radius:999px;padding:0.4rem 1.1rem;font-size:0.82rem;color:var(--fg);font-weight:600;">
                    {total_cities} Kota
                </div>
                <div style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.35); border-radius:999px;padding:0.4rem 1.1rem;font-size:0.82rem;color:var(--fg);font-weight:600;">
                    {total_categories} Kategori
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True,
    )

    st.markdown('<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;">Alur Kerja (Workflow)</div>', unsafe_allow_html=True)

    steps = [
        ("📂", "Dataset",       "#3b82f6", "rgba(59,130,246,0.12)",  "Melihat dataset mentah pariwisata yang dikumpulkan."),
        ("📊", "EDA",           "#8b5cf6", "rgba(139,92,246,0.12)", "Eksplorasi visual persebaran kota, kategori, harga, dan rating."),
        ("⚙️", "Preprocessing", "#f59e0b", "rgba(245,158,11,0.12)", "Membersihkan NaN, hapus kolom tidak relevan, & normalisasi teks."),
        ("🧠", "Training",      "#10b981", "rgba(16,185,129,0.12)", "Vektorisasi teks menggunakan TF-IDF dan hitung Cosine Similarity."),
        ("🔮", "Rekomendasi",   "#ec4899", "rgba(236,72,153,0.12)", "Cari destinasi wisata mirip atau filter berdasarkan preferensi Anda."),
    ]

    cols = st.columns(len(steps))
    for col, (icon, title, accent, bg, desc) in zip(cols, steps):
        col.markdown(
            f"""
            <div style="background:{bg};border:1px solid {accent}40;border-top:3px solid {accent};border-radius:10px;padding:1.1rem 1rem 1rem 1rem;min-height:160px;display:flex;flex-direction:column;margin-bottom:1rem;">
                <div style="font-size:1.6rem;margin-bottom:0.5rem;">{icon}</div>
                <div style="font-size:1.1rem;font-weight:700;color:var(--fg);margin-bottom:0.35rem;letter-spacing:-0.01em;">{title}</div>
                <div style="font-size:0.88rem;color:var(--fg-2);line-height:1.5;flex:1;">{desc}</div>
            </div>
            """, unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Dataset":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">📂 Raw Dataset</div>
        <div class="page-header-purpose">Eksplorasi struktur dataset mentah sebelum pemrosesan</div>
        <div class="page-header-steps">Step 1 of 5</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(f'<div class="metric-card"><div class="metric-label">Total Baris</div><div class="metric-value">{df_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
    col_b.markdown(f'<div class="metric-card"><div class="metric-label">Total Kolom</div><div class="metric-value">{df_raw.shape[1]}</div></div>', unsafe_allow_html=True)
    col_c.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values (Total)</div><div class="metric-value">{df_raw.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f4 = st.columns([2, 4, 1])
    with f1:
        kota_list = sorted(df_raw["City"].dropna().unique().tolist()) if "City" in df_raw.columns else []
        kota_f = st.selectbox("Filter berdasarkan Kota", ["Semua"] + kota_list, key="ds_city")
    with f2:
        cols_sel = st.multiselect("Pilih Kolom", df_raw.columns.tolist(), default=df_raw.columns.tolist()[:7], key="ds_cols")
    with f4:
        n_rows = st.number_input("Baris ditampilkan", min_value=10, max_value=500, value=100, step=50, key="ds_n")

    view = df_raw[cols_sel] if cols_sel else df_raw
    if kota_f != "Semua" and "City" in view.columns:
        view = view[view["City"] == kota_f]
        
    st.dataframe(view.head(n_rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "EDA":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">📊 Exploratory Data Analysis</div>
        <div class="page-header-purpose">Visualisasi distribusi data, kategori wisata, rating, dan korelasi antar fitur.</div>
        <div class="page-header-steps">Step 2 of 5</div>
    </div>
    """, unsafe_allow_html=True)

    fa, fb, fc = st.columns([2, 2, 3])
    with fa:
        city_options = ["Semua Kota"] + sorted(df_clean["City"].unique())
        eda_city = st.selectbox("Kota", city_options, key="eda_city")
    with fb:
        cat_options = ["Semua Kategori"] + sorted(df_clean["Category"].unique())
        eda_cat = st.selectbox("Kategori", cat_options, key="eda_cat")
    with fc:
        r_min, r_max = float(df_clean["Rating"].min()), float(df_clean["Rating"].max())
        eda_rating = st.slider("Range Rating", r_min, r_max, (r_min, r_max), step=0.1, key="eda_rating")

    df_e = df_clean[df_clean["Rating"].between(*eda_rating)].copy()
    if eda_city != "Semua Kota":
        df_e = df_e[df_e["City"] == eda_city]
    if eda_cat != "Semua Kategori":
        df_e = df_e[df_e["Category"] == eda_cat]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Destinasi Tampil</div><div class="metric-value">{df_e.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Kategori Tampil</div><div class="metric-value">{df_e["Category"].nunique()}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Rating</div><div class="metric-value">⭐ {df_e["Rating"].mean():.2f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Harga</div><div class="metric-value">Rp {df_e["Price"].mean():,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if df_e.empty:
        st.warning("Tidak ada data yang sesuai filter saat ini.")
    else:
        left_col, right_col = st.columns(2)

        with left_col:
            with st.expander("📊 Distribusi Kategori (Bar Chart)", expanded=True):
                st.markdown('<div class="chart-title">Jumlah Wisata per Kategori</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sns.countplot(y='Category', data=df_e, order=df_e['Category'].value_counts().index, palette='viridis', ax=ax)
                ax.set_xlabel("Jumlah Tempat Wisata")
                ax.set_ylabel("Kategori")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
            with st.expander("💰 Distribusi Harga Tiket", expanded=True):
                st.markdown('<div class="chart-title">Sebaran Harga Tiket Masuk</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sns.histplot(df_e["Price"], bins=30, kde=True, ax=ax, color=ACCENT, alpha=0.55)
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1000:.0f}k"))
                ax.set_xlabel("Harga (Rupiah)")
                ax.set_ylabel("Frekuensi")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with right_col:
            with st.expander("🏙️ Distribusi Kota (Pie Chart)", expanded=True):
                st.markdown('<div class="chart-title">Proporsi Destinasi Tiap Kota</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                tc = df_e["City"].value_counts()
                wedge_colors = sns.color_palette('magma', len(tc))
                wedges, texts, autotexts = ax.pie(
                    tc, labels=tc.index, autopct="%1.1f%%", colors=wedge_colors,
                    startangle=90, textprops={"color": TEXT_COLOR, "fontsize": 8},
                    wedgeprops={"linewidth": 1.5, "edgecolor": CHART_BG}, pctdistance=0.78
                )
                for at in autotexts:
                    at.set_color("white")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with st.expander("⭐ Rating vs Harga (Boxplot)", expanded=True):
                st.markdown('<div class="chart-title">Sebaran Rating Berdasarkan Kategori</div>', unsafe_allow_html=True)
                fig, ax = styled_fig(7, 4)
                sns.boxplot(x='Rating', y='Category', data=df_e, palette='Set2', ax=ax)
                ax.set_xlabel("Rating")
                ax.set_ylabel("Kategori")
                apply_chart_style(fig, ax)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Preprocessing":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">⚙️ Data Preprocessing & Cleaning</div>
        <div class="page-header-purpose">Proses Data Cleaning dan pembuatan Text Normalization untuk Content-Based Filtering.</div>
        <div class="page-header-steps">Step 3 of 5</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Total Raw Data</div><div class="metric-value">{df_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Setelah Cleaning</div><div class="metric-value">{df_clean.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Kolom Dihapus</div><div class="metric-value">6</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Langkah Preprocessing:</b><br>
    1. <b>Hapus kolom tidak relevan:</b> <code>Time_Minutes, Coordinate, Lat, Long, Unnamed: 11, Unnamed: 12</code>.<br>
    2. <b>Hapus Baris Null (Missing Values)</b> pada kolom target.<br>
    3. <b>Text Normalization:</b> Lowercase, hapus simbol & angka dengan Regular Expression (Regex).<br>
    4. <b>Feature Merging:</b> Membuat kolom <code>tags</code> yang berisi gabungan dari Deskripsi, Kategori (dikalikan 2), dan Kota (dikalikan 2) untuk penekanan bobot.
    </div>
    """, unsafe_allow_html=True)

    missing = df_raw.isnull().sum()
    missing = missing[missing > 0]
    
    col_chart, col_preview = st.columns([1, 2])
    with col_chart:
        st.markdown('<div class="chart-title">Missing Values (Raw Data)</div>', unsafe_allow_html=True)
        if missing.empty:
            st.success("Tidak ada missing values")
        else:
            fig, ax = styled_fig(4, max(2.5, len(missing) * 0.5))
            cols_m = bar_colors(len(missing), base="#f59e0b")
            missing.sort_values().plot(kind="barh", ax=ax, color=cols_m, zorder=2)
            add_bar_labels(ax, ax.patches, fmt="{:.0f}", color=TEXT_COLOR)
            apply_chart_style(fig, ax)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
    with col_preview:
        st.markdown('<div class="chart-title">Contoh Fitur \'tags\' (Setelah Text Cleaning)</div>', unsafe_allow_html=True)
        st.dataframe(df_clean[['Place_Name', 'tags']].head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING / EVALUASI
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Training":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">🧠 Model Training & Evaluasi</div>
        <div class="page-header-purpose">Menggunakan TF-IDF Vectorizer dan Cosine Similarity untuk menghasilkan rekomendasi.</div>
        <div class="page-header-steps">Step 4 of 5</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Algoritma (Content-Based Filtering)")
        st.markdown("""
        <div class="info-box">
        <b>TF-IDF (Term Frequency-Inverse Document Frequency)</b><br><br>
        Menghitung seberapa penting suatu kata dalam suatu dokumen (deskripsi tempat wisata). Kata yang sering muncul di satu destinasi tetapi jarang di tempat lain akan memiliki bobot tinggi.
        <br><br>
        <b>Cosine Similarity</b><br><br>
        Menghitung sudut (kemiripan) antara dua vektor destinasi wisata dari hasil matriks TF-IDF. Nilai mendekati 1 artinya sangat mirip.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Baseline Model (Top 5 Destinasi Populer)")
        baseline = df_clean.sort_values(by=['Rating', 'Price'], ascending=[False, True]).head(5)
        st.dataframe(baseline[['Place_Name', 'Category', 'City', 'Rating', 'Price']], use_container_width=True)

    with right:
        st.markdown("#### Matriks & Performa")
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">Dimensi TF-IDF Matrix</div><div class="metric-value" style="font-size:1.4rem">{tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]}</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">Dimensi Cosine Sim</div><div class="metric-value" style="font-size:1.4rem">{cosine_sim.shape[0]} x {cosine_sim.shape[1]}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>Hasil Evaluasi Keseluruhan Model (@k=5)", unsafe_allow_html=True)
        em1, em2, em3, em4 = st.columns(4)
        em1.markdown('<div class="metric-card" style="padding:0.5rem"><div class="metric-label">Mean Prec</div><div class="metric-value" style="font-size:1.2rem">0.8412</div></div>', unsafe_allow_html=True)
        em2.markdown('<div class="metric-card" style="padding:0.5rem"><div class="metric-label">Mean Recall</div><div class="metric-value" style="font-size:1.2rem">0.8561</div></div>', unsafe_allow_html=True)
        em3.markdown('<div class="metric-card" style="padding:0.5rem"><div class="metric-label">F1-Score</div><div class="metric-value" style="font-size:1.2rem">0.8462</div></div>', unsafe_allow_html=True)
        em4.markdown('<div class="metric-card" style="padding:0.5rem"><div class="metric-label">MAP</div><div class="metric-value" style="font-size:1.2rem">0.8976</div></div>', unsafe_allow_html=True)
        
        st.caption("Berdasarkan evaluasi relevansi Kategori antara target dan hasil rekomendasi (seperti di Jupyter Notebook).")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION / REKOMENDASI (MENGGUNAKAN NATIVE STREAMLIT)
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Prediction":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">🔮 Sistem Rekomendasi Pintar</div>
        <div class="page-header-purpose">Dapatkan saran destinasi terbaik berdasarkan minat Anda atau filter kota/kategori.</div>
        <div class="page-header-steps">Step 5 of 5</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Cara kerja:** Jika Anda memasukkan 'Nama Tempat / Keyword', sistem akan mencari kecocokan nama wisata (Cosine Similarity) atau keyword. Jika dikosongkan, sistem akan memfilter berdasarkan Kota dan Kategori yang dipilih.")

    with st.form("recommendation_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_nama = st.text_input("1. Nama Tempat / Keyword", placeholder="Contoh: Borobudur / Pantai")
        with c2:
            input_kota = st.selectbox("2. Pilih Kota", ["Semua"] + valid_cities)
        with c3:
            input_kategori = st.selectbox("3. Pilih Kategori", ["Semua"] + valid_categories)
            
        top_k = st.slider("Jumlah Rekomendasi", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("✨ Tampilkan Rekomendasi")

    if submitted:
        kw_param = input_nama.strip() if input_nama.strip() != "" else None
        kota_param = input_kota if input_kota != "Semua" else None
        kat_param = input_kategori if input_kategori != "Semua" else None
        
        with st.spinner("Mencari destinasi terbaik..."):
            filtered_df = df_clean.copy()
            
            if kw_param:
                idx_list = filtered_df[filtered_df['Place_Name'].str.lower() == kw_param.lower()].index
                
                if len(idx_list) > 0:
                    idx = idx_list[0]
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[0:top_k+1]
                    place_indices = [i[0] for i in sim_scores]
                    
                    recomends = filtered_df.iloc[place_indices].copy()
                    recomends['Similarity'] = [round(i[1], 2) for i in sim_scores]
                    filtered_df = recomends
                else:
                    filtered_df = filtered_df[filtered_df['tags'].str.contains(kw_param.lower(), na=False)]
            
            if 'Similarity' not in filtered_df.columns:
                if kota_param:
                    filtered_df = filtered_df[filtered_df['City'].str.lower() == kota_param.lower()]
                if kat_param:
                    filtered_df = filtered_df[filtered_df['Category'].str.lower() == kat_param.lower()]
                
                filtered_df = filtered_df.sort_values(by=['Rating', 'Price'], ascending=[False, True]).head(top_k)

            # --- TAMPILAN HASIL MENGGUNAKAN NATIVE STREAMLIT ---
            st.markdown("---")
            if filtered_df.empty:
                st.error("Maaf, tidak ada wisata yang sesuai kriteria Anda.")
            else:
                st.success(f"Ditemukan {len(filtered_df)} rekomendasi destinasi wisata!")
                
                for i, row in filtered_df.iterrows():
                    sim_text = f" &nbsp;|&nbsp; 🎯 **Similarity:** {row['Similarity']*100:.0f}%" if 'Similarity' in row else ""
                    
                    with st.container(border=True):
                        st.subheader(f"🏯 {row['Place_Name']}")
                        st.markdown(f"📍 **Kota:** {row['City']} &nbsp;|&nbsp; 🏷️ **Kategori:** {row['Category']} &nbsp;|&nbsp; ⭐ **Rating:** {row['Rating']}{sim_text}")
                        st.markdown(f"💰 **Harga Tiket:** Rp {row['Price']:,}")
                        st.write(row['Description'])

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT US
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "About Us":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">👥 About Us</div>
        <div class="page-header-purpose">Tim Pengembang Sistem Rekomendasi Destinasi Wisata Indonesia</div>
        <div class="page-header-steps">Step 6 of 6</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(59,130,246,0.1) 0%,rgba(139,92,246,0.1) 100%); padding:2rem; border-radius:12px; text-align:center; border:1px solid rgba(59,130,246,0.2); margin-bottom:2rem;">
        <div style="font-size:1.8rem; font-weight:800; color:var(--fg);">Sistem Rekomendasi Pariwisata ML</div>
        <div style="font-size:1rem; color:var(--fg-2);">Implementasi Content-Based Filtering menggunakan TF-IDF & Cosine Similarity</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>Tentang Proyek Ini</h4>
        Aplikasi web ini dibangun untuk merekomendasikan destinasi wisata kepada wisatawan domestik dan internasional di Indonesia. 
        Sistem memanfaatkan teknik pemrosesan bahasa alami (NLP) untuk menganalisis deskripsi, kategori, dan lokasi tempat wisata.
    </div>
    <div class="info-box">
        <h4>Teknologi yang Digunakan</h4>
        Python &nbsp;·&nbsp; Scikit-learn (TF-IDF, Cosine Similarity) &nbsp;·&nbsp; Pandas &nbsp;·&nbsp;
        Matplotlib / Seaborn &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; NLTK
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Meet the Team")

    members = [
        ("Justin Christian Kenan / 2802399463", "Developer & Data Scientist", "👨‍💻"),
        ("Kian Aurelio Wibowo / 2802464582", "UI/UX & Frontend (Streamlit)", "🧑‍🎨"),
        ("Justin Christroper /  2802420100", "Data Engineer (Preprocessing)", "🕵️‍♂️")
    ]

    cols = st.columns(3)
    for col, (name, role, icon) in zip(cols, members):
        col.markdown(f"""
        <div class="member-card">
            <div class="member-avatar">{icon}</div>
            <div class="member-name">{name}</div>
            <div class="member-id">{role}</div>
        </div>
        """, unsafe_allow_html=True)