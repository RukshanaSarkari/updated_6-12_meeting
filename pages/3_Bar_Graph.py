import pandas as pd
import streamlit as st
from kernel import data_utils as du
from kernel import app_utils as au
from kernel.persist import load_widget_state

load_widget_state()

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
# ----------------------------------------------------------------

with st.sidebar:
    au.display_csv_selector()

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------
# ----------------------------------------------------------------
if "loaded_df" in st.session_state:

    df = st.session_state["loaded_df"]

    # Set up Streamlit app
    st.title("Bar Graph Analysis")

    # df_csv = st.session_state["csv_df"]
    # dir = st.session_state['data_dir']

    # # File upload
    # selected_file = st.selectbox(
    #     'Select a Data Set',
    #     df_csv.loc[:,'CSV collection']
    # )
    #
    # if selected_file is not None:
    #     # Read CSV file into a pandas DataFrame
    #     df = pd.read_csv(selected_file)

    # Select x-axis column and elements
    x_column = st.selectbox("Select X-axis column", options=df.columns)
    x_elements = st.multiselect("Select X-axis elements", options=df[x_column].unique())

    # Select y-axis column and elements
    y_column = st.selectbox("Select Y-axis column", options=df.columns)
    y_elements = st.multiselect("Select Y-axis elements", options=df[y_column].unique())

    # Filter data based on selected elements
    filtered_data = df[(df[x_column].isin(x_elements)) & (df[y_column].isin(y_elements))]

    # Check if data is available for chart generation
    if len(filtered_data) > 0:
        # Create bar chart
        st.bar_chart(filtered_data[[x_column, y_column]])
    else:
        st.warning("No results found for data, please select new data.")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("Table View", "Histogram")