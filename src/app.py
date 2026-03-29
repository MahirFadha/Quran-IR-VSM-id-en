import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quran VSM Search",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* Global text color to light because of the dark background */
.stApp, .stApp p, .stApp label, [data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.9) !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(212, 175, 55, 0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(212,175,55,0.12) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Amiri', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #D4AF37;
    margin: 0;
    text-shadow: 0 0 30px rgba(212,175,55,0.4);
    letter-spacing: 2px;
}
.hero-sub {
    color: rgba(255,255,255,0.7);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.badge-row {
    display: flex; gap: 0.5rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap;
}
.badge {
    background: rgba(212,175,55,0.15);
    border: 1px solid rgba(212,175,55,0.4);
    color: #D4AF37;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Result card */
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.2);
    border-left: 4px solid #D4AF37;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    backdrop-filter: blur(4px);
}
.result-card:hover {
    background: rgba(255,255,255,0.07);
    border-left-color: #F5E185;
    transform: translateX(4px);
}
.result-header {
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap;
}
.result-ref {
    font-family: 'Amiri', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #D4AF37;
}
.result-score {
    background: linear-gradient(135deg, #D4AF37, #F5E185);
    color: #0f0c29;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 800;
}
.result-text {
    color: rgba(255,255,255,0.88);
    font-size: 0.97rem;
    line-height: 1.7;
}

/* Score bar */
.score-bar-wrap { margin-top: 0.4rem; }
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    height: 6px;
    width: 100%;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #D4AF37, #F5E185);
    transition: width 0.5s ease;
}

/* Sidebar override */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(212,175,55,0.2);
    display: none;
}

/* Expander & Widgets - Comprehensive Dark Theme for Light Mode */
[data-testid="stExpander"] {
    background-color: transparent !important;
    border: 1px solid rgba(212, 175, 55, 0.25) !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
}

/* Fix for ALL Expander Headers */
[data-testid="stExpander"] summary {
    background-color: rgba(0, 0, 0, 0.4) !important;
    color: #D4AF37 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary:hover {
    background-color: rgba(0, 0, 0, 0.6) !important;
}

/* All Inputs & Selectboxes Container */
div[data-baseweb="input"], 
div[data-baseweb="base-input"], 
div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
    background-color: rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(212, 175, 55, 0.4) !important;
    border-radius: 10px !important;
}

/* Dropdown Menu (Popover List) */
[data-baseweb="popover"] ul {
    background-color: #16213e !important;
    border: 1px solid #D4AF37 !important;
}
[data-baseweb="popover"] li {
    background-color: transparent !important;
    color: white !important;
}
[data-baseweb="popover"] li:hover {
    background-color: rgba(212, 175, 55, 0.2) !important;
}

/* Dataframe / Table styling */
[data-testid="stDataFrame"] {
    background-color: #1a1a2e !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* Force text visibility */
div[data-testid="stTextInput"] input, 
div[data-testid="stTextArea"] textarea, 
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] * {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* Gold Labels for Widgets */
label[data-testid="stWidgetLabel"], .stSlider label, .stSelectbox label {
    color: #D4AF37 !important;
    font-weight: 700 !important;
}

/* Fix placeholder visibility */
::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

/* Fix for Browser Autofill / Recommendations turning white */
input:-webkit-autofill,
input:-webkit-autofill:hover, 
input:-webkit-autofill:focus, 
input:-webkit-autofill:active {
    -webkit-box-shadow: 0 0 0 30px #0f0c29 inset !important;
    -webkit-text-fill-color: white !important;
    transition: background-color 5000s ease-in-out 0s;
}


/* Button */
.stButton > button {
    background: linear-gradient(135deg, #D4AF37, #B8860B) !important;
    color: #0f0c29 !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(212,175,55,0.4) !important;
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 0.5rem;
    gap: 0.5rem;
    display: flex;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.6) !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    flex: 1;
    text-align: center;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #D4AF37, #B8860B) !important;
    color: #0f0c29 !important;
    box-shadow: 0 4px 15px rgba(212,175,55,0.3);
}

/* Metric card */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #D4AF37;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

/* Section heading */
.section-heading {
    color: #D4AF37;
    font-size: 1.05rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Divider */
hr { border-color: rgba(212,175,55,0.15) !important; }

/* Info box */
.info-box {
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: rgba(255,255,255,0.75);
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
# Path disesuaikan agar relatif terhadap root project dan menggunakan / untuk menghindari escape char (\r)
LANGUAGE_FILES = {
    '🇮🇩 Indonesia': 'resource/id.indonesian.txt',
    '🇬🇧 English'  : 'resource/en.sahih.txt',
}

SURAH_NAMES = {
    1:'Al-Fatihah',2:'Al-Baqarah',3:'Ali Imran',4:'An-Nisa',5:'Al-Maidah',
    6:'Al-Anam',7:'Al-Araf',8:'Al-Anfal',9:'At-Tawbah',10:'Yunus',
    11:'Hud',12:'Yusuf',13:'Ar-Rad',14:'Ibrahim',15:'Al-Hijr',
    16:'An-Nahl',17:'Al-Isra',18:'Al-Kahf',19:'Maryam',20:'Ta-Ha',
    21:'Al-Anbiya',22:'Al-Hajj',23:'Al-Muminun',24:'An-Nur',25:'Al-Furqan',
    26:'Ash-Shuara',27:'An-Naml',28:'Al-Qasas',29:'Al-Ankabut',30:'Ar-Rum',
    31:'Luqman',32:'As-Sajdah',33:'Al-Ahzab',34:'Saba',35:'Fatir',
    36:'Ya-Sin',37:'As-Saffat',38:'Sad',39:'Az-Zumar',40:'Ghafir',
    41:'Fussilat',42:'Ash-Shuraa',43:'Az-Zukhruf',44:'Ad-Dukhan',45:'Al-Jathiyah',
    46:'Al-Ahqaf',47:'Muhammad',48:'Al-Fath',49:'Al-Hujurat',50:'Qaf',
    51:'Adh-Dhariyat',52:'At-Tur',53:'An-Najm',54:'Al-Qamar',55:'Ar-Rahman',
    56:'Al-Waqiah',57:'Al-Hadid',58:'Al-Mujadila',59:'Al-Hashr',60:'Al-Mumtahanah',
    61:'As-Saf',62:'Al-Jumuah',63:'Al-Munafiqun',64:'At-Taghabun',65:'At-Talaq',
    66:'At-Tahrim',67:'Al-Mulk',68:'Al-Qalam',69:'Al-Haqqah',70:'Al-Maarij',
    71:'Nuh',72:'Al-Jinn',73:'Al-Muzzammil',74:'Al-Muddaththir',75:'Al-Qiyamah',
    76:'Al-Insan',77:'Al-Mursalat',78:'An-Naba',79:'An-Naziat',80:'Abasa',
    81:'At-Takwir',82:'Al-Infitar',83:'Al-Mutaffifin',84:'Al-Inshiqaq',85:'Al-Buruj',
    86:'At-Tariq',87:'Al-Ala',88:'Al-Ghashiyah',89:'Al-Fajr',90:'Al-Balad',
    91:'Ash-Shams',92:'Al-Layl',93:'Ad-Duha',94:'Ash-Sharh',95:'At-Tin',
    96:'Al-Alaq',97:'Al-Qadr',98:'Al-Bayyinah',99:'Az-Zalzalah',100:'Al-Adiyat',
    101:'Al-Qariah',102:'At-Takathur',103:'Al-Asr',104:'Al-Humazah',105:'Al-Fil',
    106:'Quraysh',107:'Al-Maun',108:'Al-Kawthar',109:'Al-Kafirun',110:'An-Nasr',
    111:'Al-Masad',112:'Al-Ikhlas',113:'Al-Falaq',114:'An-Nas',
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_quran(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|', 2)
            if len(parts) == 3:
                s, a, t = parts
                data.append({'surah': int(s), 'ayat': int(a), 'teks': t.strip()})
    return pd.DataFrame(data)

@st.cache_resource(show_spinner="⏳ Membangun indeks TF-IDF untuk semua bahasa...")
def build_models():
    corpus, models_dict = {}, {}
    for lang, path in LANGUAGE_FILES.items():
        df = load_quran(path)
        df['teks_bersih'] = df['teks'].apply(preprocess)
        vec = TfidfVectorizer(min_df=2, max_df=0.95, sublinear_tf=True)
        mat = vec.fit_transform(df['teks_bersih'])
        corpus[lang] = df
        models_dict[lang] = {'vectorizer': vec, 'matrix': mat}
    return corpus, models_dict

corpus, models_data = build_models()

def cari_ayat(query, lang, top_n=5, filter_surah=None):
    df_lang    = corpus[lang]
    vectorizer = models_data[lang]['vectorizer']
    matrix     = models_data[lang]['matrix']
    q_vec      = vectorizer.transform([preprocess(query)])

    if filter_surah and filter_surah != 0:
        idx           = df_lang[df_lang['surah'] == filter_surah].index.tolist()
        target_matrix = matrix[idx]
        target_df     = df_lang.loc[idx].reset_index(drop=True)
    else:
        target_matrix = matrix
        target_df     = df_lang.reset_index(drop=True)

    skor    = cosine_similarity(q_vec, target_matrix)[0]
    top_idx = np.argsort(skor)[::-1][:top_n]
    hasil   = target_df.iloc[top_idx][['surah', 'ayat', 'teks']].copy()
    hasil['similarity'] = skor[top_idx].round(4)
    hasil   = hasil[hasil['similarity'] > 0].reset_index(drop=True)
    return hasil

# ─────────────────────────────────────────────────────────────────────────────
#  HERO HEADER & STATS
# ─────────────────────────────────────────────────────────────────────────────
total_ayat = len(corpus[list(corpus.keys())[0]])
lang0 = list(corpus.keys())[0]
total_vocab = len(models_data[lang0]['vectorizer'].vocabulary_)

st.markdown(f"""
<div class="hero">
  <div class="hero-title">🕌 Quran Verse Search</div>
  <div class="hero-sub">Vector Space Model · TF-IDF · Cosine Similarity</div>
  <div class="badge-row">
    <span class="badge">🇮🇩 Indonesia</span>
    <span class="badge">🇬🇧 English</span>
    <span class="badge">📊 TF-IDF</span>
    <span class="badge">📐 Cosine Similarity</span>
    <span class="badge">📖 {total_ayat:,} Ayat</span>
    <span class="badge">🔤 {total_vocab:,} Vocab</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR (Kosong/Minimal)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Quran Search")
    st.caption("Semua pengaturan kini ada di dalam tab Pencarian.")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_search, tab_viz, tab_tfidf, tab_about = st.tabs([
    "🔍 Pencarian", "📊 Visualisasi", "🔑 Analisis TF-IDF", "ℹ️ Tentang"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PENCARIAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown('<div class="section-heading">🔍 Pencarian Ayat Al-Quran</div>', unsafe_allow_html=True)
    # ⚙️ PENGATURAN (Pindahan dari Sidebar)
    with st.expander("⚙️ Opsi & Filter Pencarian", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            mode = st.radio("🔍 Mode Pencarian", ["Single Language", "Cross-Language (2 Bahasa)"], horizontal=True)
            top_n = st.slider("📋 Jumlah Hasil (Top-N)", 3, 20, 5)
        with c2:
            if mode == "Single Language":
                lang_choice = st.selectbox("🌐 Bahasa", list(LANGUAGE_FILES.keys()))
            
            use_filter = st.checkbox("📌 Filter Surah Tertentu", value=False)
            filter_surah = 0
            if use_filter:
                s_options = [f"{i}: {name}" for i, name in SURAH_NAMES.items()]
                selected_s = st.selectbox("📖 Pilih Surah", s_options, index=1, key="search_s_sel")
                filter_surah = int(selected_s.split(':')[0])

    st.write("")
    
    if mode == "Single Language":
        query_input = st.text_input(
            f"Ketik kata kunci dalam bahasa {lang_choice.split(' ')[-1]}",
            placeholder="contoh: dirikanlah shalat" if "Indonesia" in lang_choice else "e.g. establish prayer",
            label_visibility="visible",
        )
        search_btn = st.button("🔍 Cari Ayat", use_container_width=True)

        # Trigger pencarian jika tombol diklik ATAU jika query diisi dan user tekan Enter
        if (search_btn or query_input) and query_input.strip():
            with st.spinner("🔄 Menghitung..."):
                hasil = cari_ayat(query_input.strip(), lang=lang_choice,
                                  top_n=top_n, filter_surah=filter_surah if use_filter else None)

            if hasil.empty:
                st.warning("❌ Tidak ditemukan ayat yang relevan. Coba gunakan kata kunci yang berbeda.")
            else:
                st.markdown(f'<div class="section-heading">📖 Ditemukan {len(hasil)} Ayat Relevan</div>', unsafe_allow_html=True)
                max_score = hasil['similarity'].max()
                for _, row in hasil.iterrows():
                    pct = int((row['similarity'] / max_score) * 100) if max_score > 0 else 0
                    sname = SURAH_NAMES.get(row['surah'], f"Surah {row['surah']}")
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-ref">📖 {sname} ({row['surah']}:{row['ayat']})</span>
                            <span class="result-score">{row['similarity']:.4f}</span>
                        </div>
                        <div class="result-text">{row['teks']}</div>
                        <div class="score-bar-wrap">
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{pct}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("📋 Lihat sebagai Tabel"):
                    display_df = hasil.copy()
                    display_df['surah_name'] = display_df['surah'].map(SURAH_NAMES)
                    st.dataframe(display_df[['surah', 'surah_name', 'ayat', 'teks', 'similarity']],
                                 use_container_width=True, hide_index=True)
        elif search_btn:
            st.warning("⚠️ Masukkan kata kunci terlebih dahulu.")

    else:  # Cross-Language
        col_id, col_en = st.columns(2)
        with col_id:
            st.markdown("**🇮🇩 Query Indonesia**")
            query_id = st.text_input("", placeholder="cth: dirikanlah shalat", key="q_id", label_visibility="collapsed")
        with col_en:
            st.markdown("**🇬🇧 Query English**")
            query_en = st.text_input("", placeholder="e.g. establish prayer", key="q_en", label_visibility="collapsed")

        search_btn2 = st.button("🔍 Cari di Kedua Bahasa", use_container_width=True)

        if (search_btn2 or query_id or query_en) and (query_id.strip() or query_en.strip()):
            queries = {}
            if query_id.strip(): queries['🇮🇩 Indonesia'] = query_id.strip()
            if query_en.strip(): queries['🇬🇧 English']   = query_en.strip()

            if not queries:
                st.warning("⚠️ Masukkan minimal satu query.")
            else:
                cols = st.columns(len(queries))
                for col, (lang, q) in zip(cols, queries.items()):
                    with col:
                        st.markdown(f'<div class="section-heading">{lang}</div>', unsafe_allow_html=True)
                        with st.spinner(f"Mencari di {lang}..."):
                            hasil = cari_ayat(q, lang=lang, top_n=top_n,
                                              filter_surah=filter_surah if use_filter else None)
                        if hasil.empty:
                            st.warning("Tidak ada hasil.")
                        else:
                            max_s = hasil['similarity'].max()
                            for _, row in hasil.iterrows():
                                pct = int((row['similarity'] / max_s) * 100) if max_s > 0 else 0
                                sname = SURAH_NAMES.get(row['surah'], '')
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="result-header">
                                        <span class="result-ref">📖 {sname} ({row['surah']}:{row['ayat']})</span>
                                        <span class="result-score">{row['similarity']:.4f}</span>
                                    </div>
                                    <div class="result-text">{row['teks']}</div>
                                    <div class="score-bar-wrap">
                                        <div class="score-bar-bg">
                                            <div class="score-bar-fill" style="width:{pct}%"></div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: VISUALISASI
# ══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown('<div class="section-heading">📊 Visualisasi Cosine Similarity</div>', unsafe_allow_html=True)
    v_lang  = st.selectbox("Bahasa", list(LANGUAGE_FILES.keys()), key="v_lang")
    v_placeholder = "contoh: surga dan neraka" if "Indonesia" in v_lang else "e.g. paradise and hell"
    # Menggunakan key dinamis agar input ter-reset jika bahasa berubah
    v_query = st.text_input("Query untuk divisualisasi", placeholder=v_placeholder, key=f"v_query_{v_lang}")
    v_topn  = st.slider("Jumlah Ayat", 5, 20, 10, key="v_topn")
    btn_viz = st.button("📊 Buat Visualisasi", use_container_width=True)

    if (btn_viz or v_query) and v_query.strip():
        if v_query.strip():
            with st.spinner("Menghitung..."):
                hasil_v = cari_ayat(v_query.strip(), lang=v_lang, top_n=v_topn)
            if hasil_v.empty:
                st.warning("Tidak ada hasil.")
            else:
                fig, ax = plt.subplots(figsize=(9, max(4, v_topn * 0.5)))
                fig.patch.set_facecolor('#0f0c29')
                ax.set_facecolor('#1a1a2e')

                labels = [f"Q{r['surah']}:{r['ayat']} {SURAH_NAMES.get(r['surah'],'')}" for _, r in hasil_v.iterrows()]
                skor   = hasil_v['similarity'].tolist()
                colors = plt.cm.YlOrBr(np.linspace(0.4, 0.9, len(skor)))
                bars   = ax.barh(labels[::-1], skor[::-1], color=colors, edgecolor='none', height=0.65)

                for bar, val in zip(bars, skor[::-1]):
                    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                            f'{val:.4f}', va='center', ha='left', fontsize=9,
                            color='#D4AF37', fontweight='bold')

                ax.set_xlabel('Cosine Similarity Score', color='#D4AF37', fontsize=10)
                ax.set_title(f'Top-{v_topn} Ayat Paling Relevan\n"{v_query}"',
                             color='#D4AF37', fontsize=11, fontweight='bold', pad=15)
                ax.tick_params(colors='#cccccc', labelsize=8)
                # Gunakan tuple RGBA (0-1) karena set_color tidak menerima string rgba() CSS
                ax.spines[:].set_color((212/255, 175/255, 55/255, 0.4))
                ax.set_xlim(0, max(skor) * 1.3)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2a2a4a')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Masukkan query terlebih dahulu.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANALISIS TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
with tab_tfidf:
    st.markdown('<div class="section-heading">🔑 Analisis Bobot TF-IDF per Ayat</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Lihat kata-kata dengan bobot TF-IDF tertinggi di setiap ayat — ini adalah kata "kunci" yang membedakan ayat tersebut dari ayat lainnya.</div>', unsafe_allow_html=True)
    st.write("")

    tc1, tc2, tc3 = st.columns(3)
    with tc1: tf_lang  = st.selectbox("Bahasa", list(LANGUAGE_FILES.keys()), key="tf_lang")
    with tc2: 
        s_opts_tf = [f"{i}: {name}" for i, name in SURAH_NAMES.items()]
        sel_s_tf = st.selectbox("Surah", s_opts_tf, index=1, key="tf_s_sel")
        tf_surah = int(sel_s_tf.split(':')[0])
    with tc3: tf_ayat  = st.number_input("Ayat", 1, 300, 255, key="tf_ayat")
    tf_topn = st.slider("Top-N Kata", 5, 20, 10, key="tf_topn")

    if st.button("🔑 Analisis TF-IDF", use_container_width=True):
        df_lang    = corpus[tf_lang]
        vectorizer = models_data[tf_lang]['vectorizer']
        matrix     = models_data[tf_lang]['matrix']
        baris      = df_lang[(df_lang['surah'] == tf_surah) & (df_lang['ayat'] == tf_ayat)]

        if baris.empty:
            st.error(f"Ayat {tf_surah}:{tf_ayat} tidak ditemukan di [{tf_lang}]!")
        else:
            idx   = baris.index[0]
            fitur = vectorizer.get_feature_names_out()
            bobot = matrix[idx].toarray()[0]
            df_tf = pd.DataFrame({'Kata': fitur, 'Bobot TF-IDF': bobot})
            df_tf = df_tf[df_tf['Bobot TF-IDF'] > 0].sort_values('Bobot TF-IDF', ascending=False).head(tf_topn)

            sname = SURAH_NAMES.get(tf_surah, '')
            st.success(f"📖 {sname} ({tf_surah}:{tf_ayat}) — {baris['teks'].values[0]}")

            fig2, ax2 = plt.subplots(figsize=(8, max(3, tf_topn * 0.4)))
            fig2.patch.set_facecolor('#0f0c29')
            ax2.set_facecolor('#1a1a2e')
            colors2 = plt.cm.plasma(np.linspace(0.3, 0.85, len(df_tf)))
            ax2.barh(df_tf['Kata'][::-1], df_tf['Bobot TF-IDF'][::-1], color=colors2, height=0.6)
            ax2.set_xlabel('Bobot TF-IDF', color='#D4AF37')
            ax2.set_title(f'Top-{tf_topn} Kata — Surah {tf_surah}:{tf_ayat}', color='#D4AF37', fontweight='bold')
            ax2.tick_params(colors='#cccccc', labelsize=9)
            for spine in ax2.spines.values(): spine.set_edgecolor('#2a2a4a')
            plt.tight_layout()
            st.pyplot(fig2)

            with st.expander("📋 Lihat Tabel Lengkap"):
                st.dataframe(df_tf.reset_index(drop=True), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: TENTANG
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="section-heading">ℹ️ Tentang Proyek</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
        <b>🎓 Tugas Mata Kuliah</b><br><br>
        Implementasi <b>Information Retrieval</b> menggunakan <b>Vector Space Model (VSM)</b>
        dengan pembobotan <b>TF-IDF</b> dan pengukuran kemiripan <b>Cosine Similarity</b>
        pada dataset terjemahan Al-Quran multi-bahasa.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
        <b>📊 Dataset</b><br><br>
        • <b>6,236 ayat</b> dari 114 surah<br>
        • 🇮🇩 Terjemahan Kementerian Agama RI<br>
        • 🇬🇧 Sahih International (English)<br>
        • Sumber: <b>Tanzil.net</b>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div class="info-box">
    <b>📚 Cara Kerja VSM (Vector Space Model)</b><br><br>
    <b>1. Preprocessing:</b> Setiap ayat dibersihkan dari karakter non-huruf dan diubah ke huruf kecil.<br>
    <b>2. TF-IDF Weighting:</b> Menghitung bobot kata berdasarkan frekuensi kemunculannya di ayat (TF) dan kelangkaannya di seluruh Al-Quran (IDF).<br>
    <b>3. Space Representation:</b> Setiap ayat dan query diwakili sebagai vektor dalam ruang dimensi tinggi (sesuai jumlah kosakata).<br>
    <b>4. Cosine Similarity:</b> Menghitung sudut antara vektor query dan vektor ayat. Semakin kecil sudutnya (skor mendekati 1.0), semakin mirip konten tersebut.<br>
    <b>5. Ranking:</b> Ayat dengan skor tertinggi ditampilkan sebagai hasil yang paling relevan.
    </div>
    """, unsafe_allow_html=True)
#  DISABLE AUTOCOMPLETE (REKOMENDASI INPUT) 
st.markdown('''
<script>
    var inputs = window.parent.document.querySelectorAll('input');
    for (var i = 0; i < inputs.length; i++) {
        inputs[i].setAttribute('autocomplete', 'off');
    }
</script>
''', unsafe_allow_html=True)
