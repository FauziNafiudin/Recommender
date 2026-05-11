import pandas as pd

def find_show_by_keyword(df: pd.DataFrame, keyword: str, context_chars: int = 100) -> tuple:
    keyword_lower = keyword.lower()
    mask = df['overview_raw'].str.lower().str.contains(keyword_lower, na=False)
    results = df[mask].copy()

    if results.empty:
        return None, f"Kata '{keyword}' tidak ditemukan di overview manapun."

    def highlight_context(text, kw, chars=100):
        idx = text.lower().find(kw)
        if idx == -1:
            return text[:chars*2] + "..." if len(text) > chars*2 else text
        start = max(0, idx - chars)
        end = min(len(text), idx + len(kw) + chars)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    results['overview_snippet'] = results['overview_raw'].apply(
        lambda x: highlight_context(x, keyword_lower, context_chars)
    )
    return results[['name', 'overview_snippet', 'vote_average', 'vote_count']], None
