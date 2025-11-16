# paste near top with imports
import re
import math
import networkx as nx
import matplotlib.pyplot as plt

# ---------- Utility functions ----------
def normalize_col_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r'[^a-z0-9]+', '_', n)
    n = re.sub(r'_{2,}', '_', n)
    return n.strip('_')

def col_name_score(a: str, b: str) -> float:
    # exact match gets 1.0, normalized match 0.9, token overlap gives partial score
    if a == b:
        return 1.0
    if normalize_col_name(a) == normalize_col_name(b):
        return 0.9
    toks_a = set(normalize_col_name(a).split('_'))
    toks_b = set(normalize_col_name(b).split('_'))
    if len(toks_a & toks_b) == 0:
        return 0.0
    # partial token overlap score:
    return 0.4 + 0.6 * (len(toks_a & toks_b) / max(len(toks_a|toks_b),1))

# ---------- Core inference function ----------
def infer_joins(tables: dict,
                min_overlap_for_fk: float = 0.6,
                min_confidence: float = 0.35,
                max_candidates_per_pair: int = 3):
    """
    tables: dict mapping table_name -> pandas.DataFrame
    returns: list of candidate joins: {
      left_table, right_table, left_col, right_col,
      left_unique_pct, right_unique_pct, overlap_pct, name_score, confidence, reason, sql
    }
    """
    results = []
    names = list(tables.keys())
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            left_name = names[i]; right_name = names[j]
            left_df = tables[left_name]
            right_df = tables[right_name]
            # summary stats
            left_n = len(left_df)
            right_n = len(right_df)
            # avoid empty tables
            if left_n == 0 or right_n == 0:
                continue
            # compute distinct sets (but limit memory by sampling if huge)
            def distinct_values(df, col, sample_limit=20000):
                ser = df[col].dropna()
                if len(ser) > sample_limit:
                    ser = ser.sample(sample_limit, random_state=1)
                return set(ser.astype(str).values)
            # iterate columns pairs
            pair_scores = []
            for lc in left_df.columns:
                for rc in right_df.columns:
                    # quick type compatibility check: both numeric or both non-numeric preferred
                    try:
                        l_ser = left_df[lc].dropna()
                        r_ser = right_df[rc].dropna()
                    except Exception:
                        continue
                    if l_ser.shape[0] == 0 or r_ser.shape[0] == 0:
                        continue
                    # name match score
                    name_sc = col_name_score(str(lc), str(rc))
                    # basic dtype check
                    l_is_num = pd.api.types.is_numeric_dtype(l_ser)
                    r_is_num = pd.api.types.is_numeric_dtype(r_ser)
                    if l_is_num != r_is_num:
                        dtype_penalty = 0.7
                    else:
                        dtype_penalty = 1.0
                    # uniqueness
                    left_unique_pct = left_df[lc].nunique(dropna=True) / max(1, left_n)
                    right_unique_pct = right_df[rc].nunique(dropna=True) / max(1, right_n)
                    # overlap: fraction of distinct left values present in right
                    try:
                        left_vals = distinct_values(left_df, lc)
                        right_vals = distinct_values(right_df, rc)
                        if len(left_vals) == 0:
                            overlap_pct = 0.0
                        else:
                            overlap_pct = len(left_vals & right_vals) / len(left_vals)
                    except Exception:
                        overlap_pct = 0.0
                    # cardinality hint: PK (left_unique_pct close to 1 and overlap large)
                    pk_hint = 1.0 if left_unique_pct > 0.9 else max(0.0, left_unique_pct)
                    fk_hint = 1.0 if overlap_pct >= min_overlap_for_fk else overlap_pct
                    # size ratio adjustment (if left much larger than right, reduce score lightly)
                    size_ratio = min(left_n, right_n) / max(left_n, right_n)
                    # confidence formula (tunable)
                    confidence = (0.45 * name_sc +
                                  0.35 * fk_hint * dtype_penalty +
                                  0.15 * pk_hint +
                                  0.05 * size_ratio)
                    # boost when exact name match and overlap decent
                    if name_sc >= 0.9 and overlap_pct > 0.4:
                        confidence = min(1.0, confidence + 0.08)
                    # reason text
                    reason_parts = []
                    if name_sc >= 0.9:
                        reason_parts.append("column names match")
                    elif name_sc > 0.4:
                        reason_parts.append("partial name similarity")
                    if overlap_pct > 0:
                        reason_parts.append(f"{overlap_pct:.0%} of left values found in right")
                    if left_unique_pct > 0.9:
                        reason_parts.append("left column looks unique (pk candidate)")
                    # build SQL sample
                    sql = f"SELECT L.*, R.* FROM \"{left_name}\" AS L JOIN \"{right_name}\" AS R ON L.\"{lc}\" = R.\"{rc}\" LIMIT 100"
                    pair_scores.append({
                        "left_table": left_name, "right_table": right_name,
                        "left_col": lc, "right_col": rc,
                        "left_unique_pct": left_unique_pct,
                        "right_unique_pct": right_unique_pct,
                        "overlap_pct": overlap_pct,
                        "name_score": name_sc,
                        "confidence": confidence,
                        "reason": "; ".join(reason_parts) if reason_parts else "value overlap",
                        "sql": sql
                    })
            # pick top candidates for this pair
            pair_scores.sort(key=lambda x: x["confidence"], reverse=True)
            for cand in pair_scores[:max_candidates_per_pair]:
                if cand["confidence"] >= min_confidence:
                    results.append(cand)
    # sort global results
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

# ---------- Streamlit UI integration ----------
def show_join_suggestions(tables):
    st.header("🔗 Suggested Joins & Data Model")
    preds = infer_joins(tables)
    if not preds:
        st.info("No strong join candidates found automatically. Consider manual join keys.")
        return
    # show top suggestions grouped by table pair
    grouped = {}
    for p in preds:
        key = (p["left_table"], p["right_table"])
        grouped.setdefault(key, []).append(p)
    for (lt, rt), items in grouped.items():
        st.subheader(f"{lt}  ↔  {rt}")
        for it in items:
            st.markdown(f"- **{it['left_col']}** → **{it['right_col']}**  —  Confidence: **{it['confidence']:.2f}**  \n  _{it['reason']}_")
            cols = st.columns([3,1])
            with cols[0]:
                st.code(it['sql'], language='sql')
            with cols[1]:
                if st.button(f"Use join {lt}.{it['left_col']} → {rt}.{it['right_col']}", key=f"use_{lt}_{rt}_{it['left_col']}_{it['right_col']}"):
                    # run sample join and show result
                    try:
                        df = st.session_state.con.execute(it['sql']).df()
                        st.write(df.head(200))
                    except Exception as e:
                        st.error(f"Error executing sample join: {e}")
    # graph visualization
    if st.checkbox("Show relationship graph"):
        G = nx.DiGraph()
        for t in tables.keys():
            G.add_node(t)
        for p in preds:
            if p['confidence'] > 0.35:
                G.add_edge(p['left_table'], p['right_table'], label=f"{p['left_col']}→{p['right_col']}", weight=p['confidence'])
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G, seed=2)
        nx.draw(G, pos, with_labels=True, node_size=1600, node_color="#9fb3ff", font_size=10, arrows=True)
        # draw edge labels
        edge_labels = {(u,v): d['label'] for u,v,d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=8)
        st.pyplot(plt)
