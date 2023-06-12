import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page
import numpy as np
import matplotlib.pyplot as plt
from kernel import data_utils as du
from kernel import app_utils as au
from kernel.persist import load_widget_state

load_widget_state()

# If the session state variables were lost (the page was refreshed?), navigate the user back to the homepage
if "csv_df" not in st.session_state:
    switch_page("Homepage")

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
# ----------------------------------------------------------------

with st.sidebar:
    au.display_csv_selector()
    st.write("---")
    au.display_variable_selector()

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------
# ----------------------------------------------------------------

if st.session_state["df"] is None:
    st.warning("Empty df table.")

else:

    st.header("Table")

    au.display_table_view(st.session_state["df"])

    st.write("---")

    st.header("Variables")

    display_df = \
        st.session_state["var_meta_dfs"][st.session_state["csv_radio"]]

    au.display_variables_view(st.session_state["df"], display_df)

    # if st.session_state["edit_data"]:
    #     st.data_editor(
    #         display_df.loc[:, ["categorical", "numeric", "ordinal", "response"]],
    #         height=37 + 35 * st.session_state["n_vars"],
    #     )
    #
    # else:
    #     st.dataframe(
    #         display_df,
    #         height=37 + 35 * st.session_state["n_vars"],
    #     )
    #
    # edit_cols = st.columns([1, 1, 2], gap="small")
    # edit_cols[0].checkbox("Edit Data", key="edit_data")
    # if st.session_state["edit_data"]:
    #     edit_cols[1].button("Save Data")

    # st.write(st.session_state["df"])
    # st.write(st.session_state["df"].value_counts())
    # val_cts = st.session_state["df"].value_counts()

    # val_cts["count"] = df.value_counts()
    # val_cts = val_cts.reset_index()
    # val_cts = val_cts.rename(columns={0: "count"})
    # st.write(val_cts)

    # fig, ax = plt.subplots()
    # ax.hist(df.value_counts(), bins=20)
    # st.pyplot(fig)
    # st.bar_chart(val_cts)

    # initialize the variable info dataframe
    # var_name = list()
    # var_count
    # var_is_cat = [False] * n_vars
    # var_df = pd.DataFrame(columns=["name", "count", "categorical", "numeric", "ordinal"])

    # initialize the metadata array
    # columns = []
    #
    # if not df.empty:
    #
    #     # Detect the categorical and numerical columns
    #     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    #     categorical_cols = list(set(list(df.columns)) - set(numeric_cols))
    #
    #     # get the numerical and categorical columns
    #     columns = du.generate_meta_data(df)
    #
    #     # convert the column information into a dataframe
    #     columns_df = pd.DataFrame(columns, columns=['column_name', 'type'])
    #
    #     # save the metadata as a csv
    #     columns_df.to_csv('data/metadata/column_type_desc.csv', index=False)
    #
    #     # update the variable information table
    #     for i in range(columns_df.shape[0]):
    #         col_name = columns_df.iloc[i]['column_name']
    #         var_info_df.loc[i, "name"] = col_name
    #         var_info_df.loc[i, "count"] = len(df.loc[:, col_name].unique())
    #         var_info_df.loc[i, "type"] = columns_df.iloc[i]['type']
    #         # var_info_df.loc[i, "unique values"] = set(df.loc[:, col_name])
    #
    #     # display the variable information
    #     st.dataframe(var_info_df, hide_index=True)
    #
    #     st.write(du.compute_metadata(df))
    #
    #     # st.write(du.compute_variable_importance(df, col_name))
    #
    #     st.header("Show redundant variables?")
    #
    #     red_cols = du.get_redundant_columns
    #     corr = df.corr(method='pearson')
    #     y_var = st.radio("Select the variable to be predicted (y)", options=corr.columns)
    #     th = st.slider("Correlation Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.01,
    #                    format='%f')  # , key=None, help=None)
    #
    #     redundant_cols = pd.Series(du.get_redundant_columns(corr, y_var, th), name="Redundant Variables")
    #     new_df = du.new_df(df, redundant_cols)
    #     st.write("Redundant Variables:", redundant_cols)
    #     st.write("Number of Columns Dropped: ", len(redundant_cols))
    #     st.write("New Data: \n", new_df.head())
    #
    #     st.session_state["df_filtered"] = df

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("Data", "Bar Graph")
