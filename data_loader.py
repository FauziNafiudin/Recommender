import pandas as pd
import re

def load_and_preprocess_data(filepath: str = "top_rated_tv.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["name"])
    df["overview"] = df["overview"].fillna("")
    
    # Simpan versi asli untuk ditampilkan nanti
    df["overview_raw"] = df["overview"].copy()
    
    # Bersihkan untuk TF-IDF
    df["overview"] = (
        df["overview"]
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df.reset_index(drop=True)