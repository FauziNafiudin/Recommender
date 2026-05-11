import streamlit as st
from data_loader import load_and_preprocess_data
from recommender import compute_content_similarity, compute_collaborative_similarity, get_hybrid_recommendations

# ----------------------------------------------
# Caching data dan similarity matrix
# ----------------------------------------------
@st.cache_data
def load_data():
    return load_and_preprocess_data("top_rated_tv.csv")

@st.cache_resource
def compute_similarities(df):
    cb_sim = compute_content_similarity(df)
    cf_sim = compute_collaborative_similarity(df)
    return cb_sim, cf_sim

# ----------------------------------------------
# UI
# ----------------------------------------------
st.set_page_config(page_title="📺 TV Show Recommender", page_icon="📺", layout="wide")
st.title("Sistem Rekomendasi TV Show (Hybrid)")
st.caption("Item-Based CF + Content-Based (TF-IDF) | Weighted Hybrid")

# Load data & similarity
with st.spinner("Memuat dataset & membangun matriks kemiripan..."):
    df = load_data()
    cb_sim, cf_sim = compute_similarities(df)

# Deretan kontrol (tanpa sidebar)
st.header("⚙️ Pengaturan")

# Baris 1: Cari & Pilih TV Show
col_search, col_select = st.columns([1, 2])
with col_search:
    keyword = st.text_input("Cari TV Show", "", placeholder="Ketik nama show...")
with col_select:
    all_shows = sorted(df["name"].tolist())
    if keyword:
        filtered_shows = [name for name in all_shows if keyword.lower() in name.lower()]
    else:
        filtered_shows = all_shows

    if not filtered_shows:
        st.warning("Tidak ada show yang cocok.")
        selected_show = None
    else:
        selected_show = st.selectbox("Pilih TV Show:", filtered_shows)

# Baris 2: Bobot & jumlah rekomendasi
col_cf, col_slider, col_cb, col_num = st.columns([0.3, 3, 0.3, 1])
with col_cf:
    st.markdown("<span style='font-weight:bold; font-size:1.1rem;'>CF</span>",
                unsafe_allow_html=True)
with col_slider:
    cf_pct = st.slider("Bobot CF (%)", 0, 100, 20, label_visibility="collapsed")
with col_cb:
    st.markdown("<span style='font-weight:bold; font-size:1.1rem;'>CB</span>",
                unsafe_allow_html=True)
with col_num:
    top_n = st.number_input("Jumlah Rekomendasi", min_value=1, max_value=50, value=10)

cb_pct = 100 - cf_pct
cf_weight = cf_pct / 100.0
cb_weight = cb_pct / 100.0
st.caption(f"CF: **{cf_pct}%**  |  CB: **{cb_pct}%**")

# Tombol generate
if st.button("🚀 Generate Rekomendasi", type="primary", use_container_width=True):
    if selected_show is None:
        st.warning("Silakan pilih TV Show terlebih dahulu.")
    else:
        result, error = get_hybrid_recommendations(
            df, cb_sim, cf_sim, selected_show, cf_weight, cb_weight, top_n
        )
        if error:
            st.error(error)
        else:
            st.success(f"✅ Top {len(result)} rekomendasi untuk *{selected_show}*")
            # Tampilkan setiap rekomendasi sebagai kartu
            for _, row in result.iterrows():
                with st.container():
                    st.subheader(row["name"])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("⭐ Rating", f"{row['vote_average']:.1f}")
                    col2.metric("📊 Similarity Score", f"{row['similarity_score']:.3f}")
                    col3.metric("🗳️ Vote Count", f"{row['vote_count']}")

                    # Overview asli (non‑regex)
                    overview = row["overview_raw"]
                    if len(overview) > 200:
                        st.write(overview[:200] + "...")
                        with st.expander("Selengkapnya"):
                            st.write(overview)
                    else:
                        st.write(overview)

                    st.divider()