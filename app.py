import streamlit as st
import pandas as pd
from data_loader import load_and_preprocess_data
from recommender import (
    compute_content_similarity,
    compute_collaborative_similarity,
    get_hybrid_recommendations,
)
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
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "rekomendasi"

# Rekomendasi
if "sort_metric" not in st.session_state:
    st.session_state.sort_metric = "similarity"
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "show_name" not in st.session_state:
    st.session_state.show_name = ""

# Pencarian keyword
if "kw_results_raw" not in st.session_state:
    st.session_state.kw_results_raw = None
if "kw_display_count" not in st.session_state:
    st.session_state.kw_display_count = 10
if "kw_sort_metric" not in st.session_state:
    st.session_state.kw_sort_metric = "rating"

# ----------------------------------------------
# UI
# ----------------------------------------------
st.set_page_config(page_title="📺 TV Show Recommender", page_icon="📺", layout="wide")
st.title("Sistem Rekomendasi TV Show (Hybrid)")
st.caption("Item-Based CF + Content-Based (TF-IDF) | Weighted Hybrid")

df = load_data()
cb_sim, cf_sim = compute_similarities(df)

# ==================== TAB SWITCHER ====================
col_tab1, col_tab2 = st.columns(2)
with col_tab1:
    if st.button(
        "🎯 Rekomendasi",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "rekomendasi" else "secondary",
    ):
        st.session_state.active_tab = "rekomendasi"
        st.rerun()
with col_tab2:
    if st.button(
        "🔍 Pencarian Keyword",
        use_container_width=True,
        type="primary" if st.session_state.active_tab == "keyword" else "secondary",
    ):
        st.session_state.active_tab = "keyword"
        st.rerun()

st.divider()

# ==================== TAB REKOMENDASI ====================
if st.session_state.active_tab == "rekomendasi":
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

    if st.button("🚀 Generate Rekomendasi", type="primary", use_container_width=True):
        if selected_show is None:
            st.warning("Silakan pilih TV Show terlebih dahulu.")
        else:
            result, error = get_hybrid_recommendations(
                df, cb_sim, cf_sim, selected_show, cf_weight, cb_weight, top_n
            )
            if error:
                st.error(error)
                st.session_state.recommendations = None
            else:
                result = result.copy()
                result["popularity"] = result["vote_average"] * result["vote_count"]
                st.session_state.recommendations = result
                st.session_state.show_name = selected_show

    if st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        show_name = st.session_state.show_name
        st.success(f"✅ Top {len(recs)} rekomendasi untuk *{show_name}*")

        st.write("**Urutkan berdasarkan:**")
        cols_btn = st.columns(3)
        # Urutan: Rating - Similarity Score - Popularity
        metrics = [
            ("rating", "Rating"),
            ("similarity", "Similarity Score"),
            ("popularity", "Popularity"),
        ]
        for col, (key, label) in zip(cols_btn, metrics):
            btn_type = "primary" if st.session_state.sort_metric == key else "secondary"
            if col.button(label, key=f"btn_{key}", use_container_width=True, type=btn_type):
                st.session_state.sort_metric = key
                st.rerun()

        sort_key = st.session_state.sort_metric
        if sort_key == "similarity":
            recs_sorted = recs.sort_values("similarity_score", ascending=False)
        elif sort_key == "rating":
            recs_sorted = recs.sort_values("vote_average", ascending=False)
        else:  # popularity
            recs_sorted = recs.sort_values("popularity", ascending=False)

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
else:
    st.subheader("🔍 Pencarian Kata Kunci di Overview")
    search_kw = st.text_input(
        "Masukkan kata kunci", placeholder="contoh: lawyer, space, family", key="search_kw"
    )

    if search_kw:
        # Cari dan simpan data mentah
        raw, error = find_show_by_keyword(df, search_kw)
        if error:
            st.warning(error)
            st.session_state.kw_results_raw = None
            st.session_state.kw_display_count = 10
        else:
            raw = raw.copy()
            raw["popularity"] = raw["vote_average"] * raw["vote_count"]
            st.session_state.kw_results_raw = raw
            st.session_state.kw_display_count = 10   # reset ke 10 item
            st.session_state.kw_sort_metric = "rating"

    # Tampilkan hasil jika ada data mentah
    if st.session_state.kw_results_raw is not None:
        data = st.session_state.kw_results_raw
        total = len(data)

        # Sorting (dua tombol seimbang)
        st.write("**Urutkan berdasarkan:**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Rating",
                key="kw_sort_rating",
                use_container_width=True,
                type="primary" if st.session_state.kw_sort_metric == "rating" else "secondary",
            ):
                st.session_state.kw_sort_metric = "rating"
                st.session_state.kw_display_count = 10
                st.rerun()
        with c2:
            if st.button(
                "Popularity",
                key="kw_sort_pop",
                use_container_width=True,
                type="primary" if st.session_state.kw_sort_metric == "popularity" else "secondary",
            ):
                st.session_state.kw_sort_metric = "popularity"
                st.session_state.kw_display_count = 10
                st.rerun()

        # Urutkan data sesuai metrik
        sorted_data = data.sort_values(
            by="vote_average" if st.session_state.kw_sort_metric == "rating" else "popularity",
            ascending=False,
        )

        # Potong sesuai display_count
        display_count = min(st.session_state.kw_display_count, total)
        page_data = sorted_data.iloc[:display_count]

        st.success(f"Ditemukan **{total}** show yang mengandung kata *'{search_kw}'*")

        for _, row in page_data.iterrows():
            with st.container():
                st.subheader(row["name"])
                col1, col2 = st.columns(2)
                col1.metric("⭐ Rating", f"{row['vote_average']:.1f}")
                col2.metric("🔥 Popularity", f"{row['popularity']:.0f}")
                snippet = row["overview_snippet"]
                st.write(snippet)
                if len(row["overview_raw"]) > len(snippet.replace("...", "")):
                    with st.expander("Selengkapnya"):
                        st.write(row["overview_raw"])
                st.divider()

        # --- Load More + input lompat ---
        if display_count < total:
            col_load, col_jump = st.columns([3, 1])
            with col_load:
                if st.button("📥 Tampilkan lebih banyak…", use_container_width=True):
                    st.session_state.kw_display_count = min(st.session_state.kw_display_count + 10, total)
                    st.rerun()
            with col_jump:
                # Input manual untuk langsung ke jumlah item tertentu
                jump_to = st.number_input(
                    "Lompat ke",
                    min_value=1,
                    max_value=total,
                    value=display_count,
                    step=10,
                    label_visibility="collapsed",
                )
                if st.button("Terapkan"):
                    st.session_state.kw_display_count = min(int(jump_to), total)
                    st.rerun()
        else:
            st.info("✅ Semua hasil sudah ditampilkan.")
