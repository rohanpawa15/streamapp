# app.py
import streamlit as st
import pandas as pd
import duckdb
import uuid
import io
import re

st.set_page_config(page_title="Auto Join Inference", layout="wide")
st.title("📁 Unified Data Explorer — Upload & Infer Table Joins")

# -------------------------
# Utility: session DB
# -------------------------
if "con" not in st.session_state:
    st.session_state.con = duckdb.connect(database=":memory:")

con = st.session_state.con

# -------------------------
# Robust file loader
# -------------------------
def robust_read_file(uploaded_file):
    """Try multiple strategies to read an uploaded file into a pandas DataFrame."""
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    # read bytes and decode safely
    raw = uploaded_file.read()
    # Try pandas readers with fallbacks
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            # try default, then python engine skip bad lines
            try:
                return pd.read_csv(io.BytesIO(raw))
            except Exception:
                try:
                    return pd.read_csv(io.BytesIO(raw), engine="python", on_bad_lines="skip")
                except Exception:
                    return pd.read_csv(io.BytesIO(raw), sep=",", engine="python", on_bad_lines="skip", encoding="utf-8", error_bad_lines=False)
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(raw))
        elif name.endswith(".json"):
            s = raw.decode("utf-8", errors="replace")
            try:
                return pd.read_json(io.StringIO(s))
            except Exception:
                try:
                    return pd.read_json(io.StringIO(s), lines=True)
                except Exception:
                    # fallback: read as records
                    import json
                    data = json.loads(s)
                    return pd.DataFrame(data)
        elif name.endswith(".parquet"):
            # parquet requires pyarrow or fastparquet in environment
            return pd.read_parquet(io.BytesIO(raw))
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Failed to parse {uploaded_file.name}: {e}")
        return None

# -------------------------
# Upload UI (sidebar)
# -------------------------
st.sidebar.header("1) Upload up to 5 files")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON / Parquet",
    type=["csv", "xls", "xlsx", "json", "parquet"],
    accept_multiple_files=True
)

if "tables" not in st.session_state:
    st.session_state.tables = {}  # mapping name -> {'df': df, 'orig_name': str}

# Clear button
if st.sidebar.button("Clear uploaded tables"):
    st.session_state.tables = {}
    st.experimental_rerun()

# handle uploads
if uploaded_files:
    uploaded_files = uploaded_files[:5]
    for f in uploaded_files:
        key_base = f.name.rsplit(".", 1)[0].replace(" ", "_")[:40]
        key = f"{key_base}_{str(uuid.uuid4())[:6]}"
        df = robust_read_file(f)
        if df is None:
            continue
        # ensure df is materialized
        df = df.copy()
        st.session_state.tables[key] = {"df": df, "orig_name": f.name}
        # register in DuckDB (re-register each run)
        try:
            con.register(key, df)
        except Exception:
            try:
                con.unregister(key)
            except Exception:
                pass
            con.register(key, df)

# If no tables, prompt user
if not st.session_state.tables:
    st.info("Upload up to 5 data files using the sidebar. Supported: CSV, Excel, JSON, Parquet (needs pyarrow).")
    st.stop()

# -------------------------
# Small preview panel
# -------------------------
st.subheader("Uploaded Tables")
cols = st.columns(3)
i = 0
for k, meta in st.session_state.tables.items():
    with cols[i % 3]:
        st.markdown(f"**{meta['orig_name']}**  —  `{k}`")
        st.write(f"rows: {meta['df'].shape[0]} | columns: {meta['df'].shape[1]}")
    i += 1

# -------------------------
# Analyze button
# -------------------------
st.markdown("---")
if st.button("🔎 Analyze tables for join relationships"):
    st.session_state.analysis_run = True

# -------------------------
# Join inference utilities
# -------------------------
def normalize_col(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r'[^a-z0-9]+', '_', n)
    n = re.sub(r'_{2,}', '_', n)
    return n.strip('_')

def col_name_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if normalize_col(a) == normalize_col(b):
        return 0.9
    toks_a = set(normalize_col(a).split('_'))
    toks_b = set(normalize_col(b).split('_'))
    if not toks_a or not toks_b:
        return 0.0
    overlap = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return 0.4 + 0.6 * (overlap / union) if overlap > 0 else 0.0

def sample_distinct_values(ser: pd.Series, limit=20000):
    ser2 = ser.dropna().astype(str)
    if ser2.shape[0] > limit:
        ser2 = ser2.sample(limit, random_state=1)
    return set(ser2.unique())

def infer_joins_simple(tables_dict, min_overlap=0.6, min_conf=0.35):
    """
    Returns list of candidate joins sorted by confidence.
    Each item: {left_table, right_table, left_col, right_col, overlap_pct, left_unique_pct, name_score, confidence, reason, sql}
    """
    results = []
    names = list(tables_dict.keys())
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            L = names[i]
            R = names[j]
            left_df = tables_dict[L]
            right_df = tables_dict[R]
            nL = len(left_df); nR = len(right_df)
            if nL == 0 or nR == 0:
                continue
            for lc in left_df.columns:
                for rc in right_df.columns:
                    try:
                        lser = left_df[lc].dropna()
                        rser = right_df[rc].dropna()
                    except Exception:
                        continue
                    if lser.shape[0] == 0 or rser.shape[0] == 0:
                        continue
                    name_sc = col_name_similarity(str(lc), str(rc))
                    # dtype relax: treat numeric vs numeric better
                    lnum = pd.api.types.is_numeric_dtype(lser)
                    rnum = pd.api.types.is_numeric_dtype(rser)
                    dtype_factor = 1.0 if lnum == rnum else 0.8
                    left_unique = left_df[lc].nunique(dropna=True) / max(1, nL)
                    right_unique = right_df[rc].nunique(dropna=True) / max(1, nR)
                    # overlap sampling
                    try:
                        left_vals = sample_distinct_values(left_df[lc])
                        right_vals = sample_distinct_values(right_df[rc])
                        overlap = (len(left_vals & right_vals) / len(left_vals)) if left_vals else 0.0
                    except Exception:
                        overlap = 0.0
                    pk_hint = 1.0 if left_unique > 0.9 else left_unique
                    fk_hint = 1.0 if overlap >= min_overlap else overlap
                    size_ratio = min(nL, nR) / max(nL, nR)
                    confidence = (0.45 * name_sc + 0.35 * fk_hint * dtype_factor + 0.15 * pk_hint + 0.05 * size_ratio)
                    if name_sc >= 0.9 and overlap > 0.4:
                        confidence = min(1.0, confidence + 0.08)
                    reason_parts = []
                    if name_sc >= 0.9:
                        reason_parts.append("column names match")
                    elif name_sc > 0.4:
                        reason_parts.append("name similarity")
                    if overlap > 0:
                        reason_parts.append(f"{overlap:.0%} left values in right")
                    if left_unique > 0.9:
                        reason_parts.append("left column looks unique (PK candidate)")
                    sql = f'SELECT L.*, R.* FROM "{L}" AS L JOIN "{R}" AS R ON L."{lc}" = R."{rc}" LIMIT 100'
                    results.append({
                        "left_table": L, "right_table": R,
                        "left_col": lc, "right_col": rc,
                        "overlap_pct": overlap,
                        "left_unique_pct": left_unique,
                        "right_unique_pct": right_unique,
                        "name_score": name_sc,
                        "confidence": confidence,
                        "reason": "; ".join(reason_parts) if reason_parts else "value overlap",
                        "sql": sql
                    })
    # filter & sort
    results = [r for r in results if r["confidence"] >= min_conf]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

# -------------------------
# Run analysis if requested
# -------------------------
if st.session_state.get("analysis_run", False):
    st.markdown("## 🔗 Suggested Joins & Data Model (inferred)")
    # normalize tables dict: name -> DataFrame
    simple_tables = {k: v["df"] if isinstance(v, dict) and "df" in v else v for k, v in st.session_state.tables.items()}
    if len(simple_tables) < 2:
        st.info("Upload at least 2 tables to infer joins.")
    else:
        # reduce thresholds for initial visibility, user can tweak later
        candidates = infer_joins_simple(simple_tables, min_overlap=0.25, min_conf=0.20)
        if not candidates:
            st.info("No confident joins found. Try lowering thresholds or inspect column names and values.")
            # show debug hint
            st.write("Tables analyzed:")
            for n, df in simple_tables.items():
                st.write(f"- `{n}`: rows={df.shape[0]}, cols={list(df.columns)[:10]}")
        else:
            # group by table pairs
            grouped = {}
            for c in candidates:
                key = (c["left_table"], c["right_table"])
                grouped.setdefault(key, []).append(c)
            for (L, R), items in grouped.items():
                st.subheader(f"{L}  ↔  {R}")
                for it in items:
                    st.markdown(f"- **{it['left_col']}** → **{it['right_col']}** — Confidence **{it['confidence']:.2f}**  \n  _{it['reason']}_")
                    cols = st.columns([3,1])
                    with cols[0]:
                        st.code(it["sql"], language="sql")
                    with cols[1]:
                        if st.button(f"Preview {L}.{it['left_col']} → {R}.{it['right_col']}", key=f"pv_{L}_{R}_{it['left_col']}_{it['right_col']}"):
                            try:
                                df_preview = con.execute(it["sql"]).df()
                                st.dataframe(df_preview.head(200))
                                st.download_button("Download preview CSV", df_preview.to_csv(index=False).encode("utf-8"), file_name="join_preview.csv")
                            except Exception as e:
                                st.error(f"Preview failed: {e}")

            # optional graph (only if networkx available)
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                G = nx.DiGraph()
                for t in simple_tables.keys():
                    G.add_node(t)
                for c in candidates:
                    if c["confidence"] > 0.25:
                        G.add_edge(c["left_table"], c["right_table"], label=f"{c['left_col']}→{c['right_col']}", weight=c["confidence"])
                plt.figure(figsize=(8, 6))
                pos = nx.spring_layout(G, seed=2)
                nx.draw(G, pos, with_labels=True, node_size=1400, node_color="#9fb3ff", font_size=10, arrows=True)
                edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray", font_size=8)
                st.pyplot(plt)
            except Exception:
                st.info("Graph visualization disabled (install networkx + matplotlib to enable).")

    # reset flag so user can re-run if needed
    st.session_state.analysis_run = False

# End
st.markdown("---")
st.caption("Notes: inference uses heuristic rules (name similarity, value overlap, uniqueness). Adjust thresholds for sensitivity.")
