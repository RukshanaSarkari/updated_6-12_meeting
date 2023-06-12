import numpy as np
import pandas as pd
import plotly.express as px
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from kernel import data_utils as du
from kernel.persist import persist


def display_csv_collection():
    # ----------------------------------------------------------------
    # CSV Selection Table

    # List all .csv files in the folder
    csv_files = du.get_files_in_folder(st.session_state['data_dir'], "csv")

    if csv_files:

        # Display CSVs in a df edit table
        if len(csv_files) > 0:
            selected = [False] * len(csv_files)
            csv_df = pd.DataFrame(data=zip(selected, csv_files), columns=["*", "CSV collection"])

            st.data_editor(
                csv_df,
                column_config={
                    "Selection": st.column_config.CheckboxColumn(
                        "",
                        default=False,
                    )
                },
                key="cvs_selector",
                hide_index=True,
                height=37 + 35 * len(csv_files),
            )

            if "edited_rows" in st.session_state["cvs_selector"]:
                edited_rows = st.session_state["cvs_selector"]["edited_rows"]

                old_csv_names = []
                new_csv_names = []
                csv_files_selected = []
                rows_selected = []

                # Scan through the edited rows and collect all the new names
                for row, val in edited_rows.items():

                    if "CSV collection" in val:
                        old_csv_names.append(f"{st.session_state['data_dir']}//{csv_files[row]}")
                        new_csv_names.append(f"{st.session_state['data_dir']}//{val['CSV collection']}")

                    if "*" in val:
                        csv_files_selected.append(f"{st.session_state['data_dir']}//{csv_files[row]}")
                        rows_selected.append(row)

        csv_action_cols = st.columns([1, 1, 1], gap="small")

        with csv_action_cols[0]:

            # Display an update csv-name button
            save_csv_button = st.button(
                "Save",
                help="Any csv-file names will be saved."
            )

            if save_csv_button:
                du.rename_csv_files(old_csv_names, new_csv_names)

        with csv_action_cols[1]:

            # Display an delete csv button
            del_csv_button = st.button(
                "Delete",
                help="Deletes all selected csv-files from the collection."
            )

            if del_csv_button:
                # Delete the files from the local df folder
                du.delete_csv_files(csv_files_selected)

                # Update the csv collection table
                csv_df.drop(rows_selected)
                st.experimental_rerun()

        st.session_state["csv_df"] = csv_df

    else:
        st.warning("No CSV files imported.")


def display_csv_selector():
    if "csv_df" in st.session_state:

        # display the csv selection radio button
        st.radio(
            "Select CSV:",
            st.session_state["csv_df"]["CSV collection"],
            key=persist("csv_radio"),
        )

        if st.session_state["csv_radio"]:
            st.session_state["loaded_df"] = st.session_state["csv_dfs"][st.session_state["csv_radio"]]


def display_variable_selector():
    # ----------------------------------------------------------------
    # Variable Selection Table
    tbl = st.session_state["loaded_df"]
    var_names = list(tbl.columns)
    n_vars = len(var_names)
    var_indices = [True] * n_vars

    var_df = pd.DataFrame(data=zip(var_indices, var_names), columns=["Selection", "Variables"])

    # Table with checkboxes
    st.data_editor(
        var_df,
        column_config={
            "Selection": st.column_config.CheckboxColumn(
                "",
                help="Select your",
            )
        },
        key="var_selector",
        disabled=["Variables"],
        hide_index=True,
        # Very hard coded. Check with other browsers and zoom levels
        height=37 + 35 * n_vars,
    )

    st.session_state["n_vars"] = n_vars
    var_df.set_index("Variables", inplace=True)

    if "edited_rows" in st.session_state["var_selector"]:
        edited_rows = st.session_state["var_selector"]["edited_rows"]
        for k, v in edited_rows.items():
            var_indices[k] = v["Selection"]

        st.session_state["df"] = tbl.iloc[:, var_indices]

        del st.session_state["var_selector"]["edited_rows"]

    # st.session_state["selected_vars"] = var_df.loc[selected, "Variables"]
    # st.session_state["vars"] = var_names
    # st.session_state["vars_selected"] = tbl.loc[:, var_names[[var_indices]]]
    #
    # return selected, st.session_state["selected_vars"]


def display_page_navigation(left_page, right_page):
    st.write("---")
    nav_cols = st.columns([2, 3, 2])
    with nav_cols[0]:
        if left_page != "":
            nav_left = st.button(f"⬅ {left_page}")
            if nav_left:
                switch_page(left_page.replace(" ", "_"))

    with nav_cols[2]:
        if right_page != "":
            nav_right = st.button(f"{right_page} ➡")
            if nav_right:
                switch_page(right_page.replace(" ", "_"))


def display_table_view(tbl):
    df_head_tab, df_stats_tab, df_full_tab, meta_tab = st.tabs(
        ["head", "statistics", "full table", "metadata"]
    )

    with df_head_tab:
        st.dataframe(tbl.head())

    with df_stats_tab:
        st.dataframe(tbl.describe())

    with df_full_tab:
        st.dataframe(tbl)

    with meta_tab:
        row = st.session_state["csv_meta_df"]["name"] == st.session_state["csv_radio"]
        st.dataframe(st.session_state["csv_meta_df"].loc[row, ["name", "size (mb)", "n_rows"]], hide_index=True)


def display_variables_view(df_tbl, meta_tbl):
    df_meta_tab, df_vals_tab, df_type_tab, df_edit_tab = st.tabs(
        ["metadata", "values", "types", "edit types"]
    )

    with df_meta_tab:
        st.dataframe(
            meta_tbl.loc[:, ["data type", "n values", "n missing", "n unique"]],
            height=37 + 35 * st.session_state["n_vars"],
        )

    with df_vals_tab:
        var_col, info_col = st.columns([1, 2])

        with var_col:
            var_sel = st.radio("Variables:", df_tbl.columns, label_visibility="hidden")

        with info_col:
            var_count_tbl = pd.DataFrame(df_tbl[var_sel].value_counts())
            var_count_tbl.reset_index(inplace=True)
            var_count_tbl.columns = [var_sel, "count"]
            var_count_tbl.sort_values(by=[var_sel], inplace=True)

            fig = px.pie(var_count_tbl,
                         values="count",
                         names=var_sel,
                         title=f'{var_sel}',
                         # height=300, width=200,
                         )
            st.plotly_chart(fig, use_container_width=True)

    with df_type_tab:
        st.dataframe(
            meta_tbl.loc[:, ["categorical", "numeric", "ordinal", "response"]],
            height=37 + 35 * st.session_state["n_vars"],
        )

    with df_edit_tab:
        st.data_editor(
            meta_tbl.loc[:, ["categorical", "numeric", "ordinal", "response"]],
            height=37 + 35 * st.session_state["n_vars"],
        )
        st.button("Save Data")


def getColumnTypes(cols):
    categorical = []
    numerical = []
    object = []
    for i in range(len(cols)):
        if cols["type"][i] == 'categorical':
            categorical.append(cols['column_name'][i])
        elif cols["type"][i] == 'numerical':
            numerical.append(cols['column_name'][i])
        else:
            object.append(cols['column_name'][i])
    return categorical, numerical, object


def makeMapDict(col):
    unique_vals = list(np.unique(col))
    unique_vals.sort()
    dict_ = {unique_vals[i]: i for i in range(len(unique_vals))}
    return dict_


def map_unique(df, colName):
    dict_ = makeMapDict(df[colName])
    cat = np.unique(df[colName])
    df[colName] = df[colName].map(dict_)
    return cat


# For redundant columns
def get_redundant_columns(corr, y: str, threshold=0.1):
    cols = corr.columns
    redundant = []
    k = 0
    for ind, c in enumerate(corr[y]):
        if c < 1 - threshold:
            redundant.append(cols[ind])
    return redundant


def new_df(df, columns2drop):
    return df.drop(columns2drop, axis='columns')


def compute_metadata(df):
    metadata = pd.DataFrame(columns=['Variable', 'Data Type', 'Count', 'Missing Values', 'Unique Values'])

    for column in df.columns:
        variable = column
        data_type = str(df[column].dtype)
        count = df[column].count()
        missing_values = df[column].isnull().sum()
        unique_values = df[column].nunique()

        metadata = metadata.append({'Variable': variable, 'Data Type': data_type, 'Count': count,
                                    'Missing Values': missing_values, 'Unique Values': unique_values},
                                   ignore_index=True)

    return metadata


def compute_variable_importance(df, target_column):
    # Split the DataFrame into features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Create a Random Forest regressor
    rf = RandomForestRegressor()

    # Fit the Random Forest model
    rf.fit(X, y)

    # Compute variable importance
    importance = pd.DataFrame({'Variable': X.columns, 'Importance': rf.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return importance


if __name__ == '__main__':
    df = {"Name": ["salil", "saxena", "for", "int"]}
    df = pd.DataFrame(df)
    print("Mapping dict: ", makeMapDict(df["Name"]))
    print("original df: ")
    print(df.head())
    pp = mapunique(df, "Name")
    print("New df: ")
    print(pp.head())
