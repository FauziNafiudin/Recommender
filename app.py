import streamlit as st
import pandas as pd
from data_loader import load_and_preprocess_data
from recommender import compute_content_similarity, compute_collaborative_similarity, get_hybrid_recommendations
from utils import find_show_by_keyword

# ----------------------------------------------
# Caching
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
# Inisialisasi session state
# ----------------------------------------------
if "sort_metric" not in st.session_state:
    st.session_state["sort_metric"] = "similarity"  # default
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "show_name" not in st.session_state:
    st.session_state["show_name"] = ""

# ----------------------------------------------
# UI
# ----------------------------------------------
st.set_page_config(page_title="📺 TV Show Recommender", page_icon="📺", layout="wide")
st.title("Sistem Rekomendasi TV Show (Hybrid)")
st.caption("Item-Based CF + Content-Based (TF-IDF) | Weighted Hybrid")

df = load_data()
cb_sim, cf_sim = compute_similarities(df)

# ==================== Tabs ====================
tab_rekomendasi, tab_keyword = st.tabs(["🎯 Rekomendasi", "🔍 Pencarian Keyword"])

# ==================== TAB REKOMENDASI ====================
with tab_rekomendasi:
    col1, col2 = st.columns([1, 2])
    with col1:
        keyword = st.text_input("Cari judul TV Show", "", placeholder="Ketik nama...")
    all_shows = sorted(df["name"].tolist())
    if keyword:
        filtered_shows = [name for name in all_shows if keyword.lower() in name.lower()]
    else:
        filtered_shows = all_shows

    with col2:
        if not filtered_shows:
            st.warning("Tidak ada show yang cocok.")
            selected_show = None
        else:
            selected_show = st.selectbox("Pilih TV Show:", filtered_shows)

    # Bobot & Jumlah Rekomendasi
    col_cf_label, col_slider, col_cb_label, col_num = st.columns([0.2, 2, 0.2, 1])
    with col_cf_label:
        st.markdown("<span style='font-weight:bold;'>CF</span>", unsafe_allow_html=True)
    with col_slider:
        cf_pct = st.slider("Bobot CF (%)", 0, 100, 20, label_visibility="collapsed")
    with col_cb_label:
        st.markdown("<span style='font-weight:bold;'>CB</span>", unsafe_allow_html=True)
    with col_num:
        top_n = st.number_input("Jumlah Rekomendasi", min_value=1, max_value=50, value=10)

    cb_pct = 100 - cf_pct
    cf_weight = cf_pct / 100.0
    cb_weight = cb_pct / 100.0
    st.caption(f"CF: **{cf_pct}%**  |  CB: **{cb_pct}%**")

    # Tombol Generate
    if st.button("🚀 Generate Rekomendasi", type="primary", use_container_width=True):
        if selected_show is None:
            st.warning("Silakan pilih TV Show terlebih dahulu.")
        else:
            result, error = get_hybrid_recommendations(
                df, cb_sim, cf_sim, selected_show, cf_weight, cb_weight, top_n
            )
            if error:
                st.error(error)
                st.session_state["recommendations"] = None
            else:
                result = result.copy()
                result["popularity"] = result["vote_average"] * result["vote_count"]
                st.session_state["recommendations"] = result
                st.session_state["show_name"] = selected_show

    # Tampilkan hasil jika ada
    if st.session_state["recommendations"] is not None:
        recs = st.session_state["recommendations"]
        show_name = st.session_state["show_name"]
        st.success(f"✅ Top {len(recs)} rekomendasi untuk *{show_name}*")

        # ----- Tombol Pemilihan Metrik Pengurutan -----
        st.write("**Urutkan berdasarkan:**")
        cols_btn = st.columns(3)
        metrics = [
            ("similarity", "Similarity Score"),
            ("rating", "Rating"),
            ("popularity", "Popularity")
        ]
        for col, (key, label) in zip(cols_btn, metrics):
            # Tentukan type button: primary jika terpilih, secondary jika tidak
            btn_type = "primary" if st.session_state["sort_metric"] == key else "secondary"
            if col.button(label, key=f"btn_{key}", use_container_width=True, type=btn_type):
                st.session_state["sort_metric"] = key
                st.rerun()  # agar perubahan langsung terlihat

        # Sorting
        sort_key = st.session_state["sort_metric"]
        if sort_key == "similarity":
            recs_sorted = recs.sort_values("similarity_score", ascending=False)
        elif sort_key == "rating":
            recs_sorted = recs.sort_values("vote_average", ascending=False)
        else:  # popularity
            recs_sorted = recs.sort_values("popularity", ascending=False)

        # Tampilkan kartu
        for _, row in recs_sorted.iterrows():
            with st.container():
                st.subheader(row["name"])
                col1, col2, col3 = st.columns(3)
                col1.metric("⭐ Rating", f"{row['vote_average']:.1f}")
                col2.metric("📊 Similarity", f"{row['similarity_score']:.3f}")
                col3.metric("🔥 Popularity", f"{row['popularity']:.0f}")
                overview = row["overview_raw"]
                if len(overview) > 200:
                    st.write(overview[:200] + "...")
                    with st.expander("Selengkapnya"):
                        st.write(overview)
                else:
                    st.write(overview)
                st.divider()

# ==================== TAB PENCARIAN KEYWORD ====================
with tab_keyword:
    st.subheader("🔍 Pencarian Kata Kunci di Overview")
    search_keyword = st.text_input("Masukkan kata kunci", placeholder="contoh: lawyer, space, family", key="search_kw")
    if search_keyword:
        kw_results, kw_error = find_show_by_keyword(df, search_keyword)
        if kw_error:
            st.warning(kw_error)
        else:
            st.success(f"Ditemukan **{len(kw_results)}** show yang mengandung kata *'{search_keyword}'*")
            st.dataframe(kw_results, use_container_width=True, hide_index=True)
