import pandas as pd

def find_show_by_keyword(df: pd.DataFrame, keyword: str, context_chars: int = 100) -> tuple[pd.DataFrame | None, str | None]:
    """Mencari show yang mengandung keyword di overview beserta cuplikan konteks."""
    keyword_lower = keyword.lower()
    mask = df['overview'].str.contains(keyword_lower, case=False, na=False)
    results = df[mask].copy()

    if len(results) == 0:
        return None, f"Kata '{keyword}' tidak ditemukan di overview manapun."

    def highlight_context(text: str, kw: str, chars: int = context_chars) -> str:
        idx = text.lower().find(kw)
        if idx == -1: return text
        start = max(0, idx - chars)
        end = min(len(text), idx + len(kw) + chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"

    results['overview_snippet'] = results['overview'].apply(lambda x: highlight_context(x, keyword_lower))
    return results[['name', 'overview_snippet', 'vote_average', 'vote_count']], None