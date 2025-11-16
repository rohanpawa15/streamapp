# app.py
import streamlit as st
import pandas as pd
import duckdb
import io
import uuid

st.set_page_config(page_title="Unified Data Explorer", layout="wide")

st.title("📊 Unified Data Explorer — Upload, Join & Query Multiple Files")
st.markdown("Upload up to 5 files (CSV, Excel, JSON, Parquet). Preview tables, run SQL queries (DuckDB), or use the Join Wizard.")

@st.cache_data
def init_conn():
    return duckdb.connect(database=':memory:')

con = init_conn()

def load_file_to_df(uploaded_file):
    name = uploaded_file.name
    try:
        if name.lower().endswith('.csv') or name.lower().endswith('.txt'):
            df = pd.read_csv(uploaded_file)
        elif name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif name.lower().endswith('.json'):
            # try json normal and json lines
            try:
                df = pd.read_json(uploaded_file)
            except ValueError:
                uploaded_file.seek(0)
                df = pd.read_json(uploaded_file, lines=True)
        elif name.lower().endswith('.parquet'):
            # requires pyarrow or fastparquet
            df = pd.read_parquet(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {name}")
            return None
        return df
    except Exception as e:
        st.error(f"Failed to read {name}: {e}")
        return None

# Sidebar: Uploads
st.sidebar.header("1) Upload files (max 5)")
uploaded = st.sidebar.file_uploader(
    "Choose files (CSV, Excel, JSON, Parquet)", type=['csv','xls','xlsx','json','parquet'], accept_multiple_files=True, help="Upload up to 5 files"
)

# Limit to 5
if uploaded and len(uploaded) > 5:
    st.sidebar.error("Please upload a maximum of 5 files.")
    uploaded = uploaded[:5]

tables = {}
if uploaded:
    st.sidebar.markdown("**Preview uploaded files**")
    for f in uploaded:
        df = load_file_to_df(f)
        if df is not None:
            # create a safe table name (no spaces, limited length)
            base = f.name.rsplit('.',1)[0]
            table_name = base.replace(' ', '_')[:40] + "_" + str(uuid.uuid4())[:8]
            tables[table_name] = {'df': df, 'orig_name': f.name}
            st.sidebar.write(f"**{f.name}** → registered as `{table_name}`")
            st.sidebar.write(f"shape: {df.shape}")

# If no uploads, show sample instruction
if not tables:
    st.info("Upload files via the left sidebar. You can try sample files or copy/paste a CSV to test.")
else:
    # Register tables in DuckDB
    for tname, meta in tables.items():
        con.register(tname, meta['df'])

    # Main layout: Tabs for Preview / Join Wizard / SQL Playground
    tab1, tab2, tab3 = st.tabs(["Table Preview", "Join Wizard", "SQL Playground"])

    with tab1:
        st.header("Table Preview")
        for tname, meta in tables.items():
            st.subheader(f"{meta['orig_name']}  —  `{tname}`")
            st.write(f"Rows: {meta['df'].shape[0]} | Columns: {meta['df'].shape[1]}")
            st.dataframe(meta['df'].head(200), use_container_width=True)
            if st.button(f"Download `{tname}` as CSV", key=f"dl_{tname}"):
                csv_bytes = meta['df'].to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {tname}.csv", csv_bytes, file_name=f"{tname}.csv")

    with tab2:
        st.header("Join Wizard — visually create a join between two tables")
        table_names = list(tables.keys())
        if len(table_names) < 2:
            st.warning("Upload at least two tables to use the Join Wizard.")
        else:
            left = st.selectbox("Left table", table_names, index=0)
            right = st.selectbox("Right table", table_names, index=1)
            left_df = tables[left]['df']
            right_df = tables[right]['df']

            st.markdown("Select join keys (columns must exist in the respective tables).")
            left_key = st.selectbox("Left key (column)", [""] + list(left_df.columns), index=0)
            right_key = st.selectbox("Right key (column)", [""] + list(right_df.columns), index=0)
            join_type = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=0)
            limit_rows = st.number_input("Preview rows", min_value=10, max_value=10000, value=200, step=10)

            if st.button("Run Join"):
                if not left_key or not right_key:
                    st.error("Please select both join keys.")
                else:
                    # build SQL to avoid ambiguous column names; prefix with table aliases
                    sql = f"""
                    SELECT *
                    FROM {left} AS L
                    {join_type.upper()} JOIN {right} AS R
                    ON L.`{left_key}` = R.`{right_key}`
                    LIMIT {int(limit_rows)}
                    """
                    try:
                        # ensure tables are registered
                        res = con.execute(sql).df()
                        st.success(f"Join returned {res.shape[0]} rows (showing up to {limit_rows})")
                        st.dataframe(res)
                        csv_bytes = res.to_csv(index=False).encode('utf-8')
                        st.download_button("Download join result as CSV", csv_bytes, file_name="join_result.csv")
                    except Exception as e:
                        st.error(f"Error running join: {e}")

    with tab3:
        st.header("SQL Playground — run arbitrary SQL across registered tables")
        st.markdown("Tables available for SQL (registered):")
        for tname, meta in tables.items():
            st.markdown(f"- `{tname}`  (from **{meta['orig_name']}**)")
        st.markdown("Example: `SELECT L.*, R.other_col FROM table1 AS L JOIN table2 AS R ON L.id = R.id LIMIT 100`")

        query = st.text_area("Enter SQL query", height=200, value="SELECT * FROM " + list(tables.keys())[0] + " LIMIT 200")
        if st.button("Run SQL"):
            try:
                result_df = con.execute(query).df()
                st.success(f"Query returned {result_df.shape[0]} rows.")
                st.dataframe(result_df)
                csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download query result as CSV", csv_bytes, file_name="query_result.csv")
            except Exception as e:
                st.error(f"SQL error: {e}")

    # Footer / housekeeping
    st.markdown("---")
    st.caption("Built with Streamlit + DuckDB. Uploaded tables are kept in memory for the session only.")
