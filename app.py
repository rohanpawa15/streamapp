# app.py (patched)
import streamlit as st
import pandas as pd
import duckdb
import uuid

st.set_page_config(page_title="Unified Data Explorer", layout="wide")

st.title("📊 Unified Data Explorer — Upload, Join & Query Multiple Files")

# Ensure a per-session duckdb connection (stored in session_state, not cached)
if "con" not in st.session_state:
    st.session_state.con = duckdb.connect(database=":memory:")

con = st.session_state.con

def load_file_to_df(uploaded_file):
    # read uploaded file into a fully materialized DataFrame
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        elif name.endswith(".json"):
            try:
                df = pd.read_json(uploaded_file)
            except ValueError:
                uploaded_file.seek(0)
                df = pd.read_json(uploaded_file, lines=True)
        elif name.endswith(".parquet"):
            # only if pyarrow present in environment
            df = pd.read_parquet(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return None
        # Force evaluation (avoid any lazy objects)
        df = df.copy()
        return df
    except Exception as e:
        st.error(f"Failed to read {uploaded_file.name}: {e}")
        return None

# Upload UI
st.sidebar.header("1) Upload files (max 5)")
uploaded = st.sidebar.file_uploader(
    "Choose files (CSV, Excel, JSON, Parquet)",
    type=["csv", "xls", "xlsx", "json", "parquet"],
    accept_multiple_files=True,
)

# Reset previous tables on new upload action (optional)
if st.sidebar.button("Clear uploaded tables"):
    st.session_state.pop("tables", None)
    st.experimental_rerun()

# Prepare tables dictionary in session state
if "tables" not in st.session_state:
    st.session_state.tables = {}

# Handle uploads
if uploaded:
    if len(uploaded) > 5:
        st.sidebar.error("Please upload a maximum of 5 files.")
        uploaded = uploaded[:5]
    for f in uploaded:
        # create stable key from filename + uuid to avoid collisions
        base = f.name.rsplit(".", 1)[0].replace(" ", "_")[:40]
        key = f"{base}_{str(uuid.uuid4())[:8]}"
        # read file into dataframe
        df = load_file_to_df(f)
        if df is None:
            continue
        st.session_state.tables[key] = {"df": df, "orig_name": f.name}
        # register in duckdb (re-registering on each run is fine)
        try:
            con.register(key, df)
        except Exception:
            # duckdb.register can raise if name exists; use unregister then register
            try:
                con.unregister(key)
            except Exception:
                pass
            con.register(key, df)

# If no tables, prompt upload
if not st.session_state.tables:
    st.info("Upload files via the left sidebar. You can upload CSV/Excel/JSON/Parquet.")
    st.stop()

# UI tabs
tab1, tab2, tab3 = st.tabs(["Table Preview", "Join Wizard", "SQL Playground"])

with tab1:
    st.header("Table Preview")
    for tname, meta in st.session_state.tables.items():
        st.subheader(f"{meta['orig_name']}  —  `{tname}`")
        st.write(f"Rows: {meta['df'].shape[0]} | Columns: {meta['df'].shape[1]}")
        st.dataframe(meta["df"].head(200), use_container_width=True)
        csv_bytes = meta["df"].to_csv(index=False).encode("utf-8")
        st.download_button(f"Download `{tname}` as CSV", csv_bytes, file_name=f"{tname}.csv")

with tab2:
    st.header("Join Wizard — visually create a join between two tables")
    names = list(st.session_state.tables.keys())
    if len(names) < 2:
        st.warning("Upload at least two tables to use the Join Wizard.")
    else:
        left = st.selectbox("Left table", names, index=0)
        right = st.selectbox("Right table", names, index=1)
        left_df = st.session_state.tables[left]["df"]
        right_df = st.session_state.tables[right]["df"]

        left_key = st.selectbox("Left key (column)", [""] + list(left_df.columns), index=0)
        right_key = st.selectbox("Right key (column)", [""] + list(right_df.columns), index=0)
        join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=0)
        preview = st.number_input("Preview rows", min_value=10, max_value=10000, value=200, step=10)

        if st.button("Run Join"):
            if not left_key or not right_key:
                st.error("Please select both join keys.")
            else:
                sql = f"""
                SELECT *
                FROM {left} AS L
                {join_type.upper()} JOIN {right} AS R
                ON L.`{left_key}` = R.`{right_key}`
                LIMIT {int(preview)}
                """
                try:
                    res = con.execute(sql).df()
                    st.success(f"Join returned {res.shape[0]} rows (showing up to {preview})")
                    st.dataframe(res)
                    st.download_button("Download join result as CSV", res.to_csv(index=False).encode("utf-8"), file_name="join_result.csv")
                except Exception as e:
                    st.error(f"Error running join: {e}")

with tab3:
    st.header("SQL Playground")
    st.markdown("Available tables:")
    for k, meta in st.session_state.tables.items():
        st.markdown(f"- `{k}` (from **{meta['orig_name']}**)")
    default = list(st.session_state.tables.keys())[0]
    query = st.text_area("SQL Query", value=f"SELECT * FROM {default} LIMIT 200", height=200)
    if st.button("Run SQL"):
        try:
            out = con.execute(query).df()
            st.success(f"Query returned {out.shape[0]} rows.")
            st.dataframe(out)
            st.download_button("Download query result as CSV", out.to_csv(index=False).encode("utf-8"), file_name="query_result.csv")
        except Exception as e:
            st.error(f"SQL error: {e}")

st.caption("Uploaded tables live in session state and DuckDB in-memory; they are cleared when session expires.")
