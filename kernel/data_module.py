import os
import sqlite3
import streamlit as st
import pandas as pd


# Returns all .db and .csv files in the directory specified by "path"
def get_csv_files_in_folder(path=None):
    if not os.path.exists(path):
        return None

    files = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            files.append(file)
    return files


# Returns all tables in a SQLite database specified by "db_path"
def get_tables_in_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    conn.close()
    return tables


@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)


def filter_table_columns():
    tbl = st.session_state["df"]
    cols = st.session_state["vars"]
    st.session_state["df"] = tbl.loc[:, cols]


def display_table(tbl):
    if tbl is not None:
        df_hdr_tab, df_stats_tab, df_tab = st.tabs(["header", "statistics", "full table"])

        with df_hdr_tab:
            st.dataframe(tbl.head())

        with df_stats_tab:
            st.dataframe(tbl.describe())

        with df_tab:
            st.dataframe(tbl)


def variable_selection(tbl):
    display_table(tbl.loc[:, sel_vars])
    # st.write(var_selections)
    # st.write(st.session_state["variables"])
    # st.dataframe(tbl.loc[:, st.session_state["variables"]])
    # filter_table_columns(var_selections)
    #
    # st.dataframe(st.session_state["df"])


def process_data_folder():
    data_dir = st.session_state["data_dir"]
    # List all .csv files in the folder
    data_files = get_csv_files_in_folder(data_dir)

    # Display CSVs in a dropdown
    if data_files:
        selected_file = st.selectbox("Select a .csv or .db file:", data_files)
    else:
        st.warning("Directory not found.")
        selected_file = None

    if selected_file:
        file_path = os.path.join(data_dir, selected_file)
        st.session_state["file_name"] = selected_file

        # Load the csv data
        return load_csv(file_path)


def process_drag_drop(files):
    st.write(files)


def get_dataframe_info(df):
    df_types = pd.DataFrame(df.dtypes)
    df_nulls = df.count()

    df_null_count = pd.concat([df_types, df_nulls], axis=1)
    df_null_count = df_null_count.reset_index()

    # Reassign column names
    col_names = ["features", "types"]
    df_null_count.columns = col_names

    return df_null_count
