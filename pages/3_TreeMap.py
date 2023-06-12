import pandas as pd
import streamlit as st
import plotly.express as px
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
st.title("Tree Map Analysis")

# df_csv = st.session_state["csv_df"]
# dir = st.session_state['data_dir']
#
# # File upload
# selected_file = st.selectbox(
#     'Select a Data Set',
#     df_csv.loc[:,'CSV collection']
# )

# if selected_file is not None:
#     # Read CSV file into a pandas DataFrame
#     df = pd.read_csv(dir + "//" + selected_file)

# Select column to remove duplicates
column_to_remove_duplicates = st.selectbox("Select column to remove duplicates", options=df.columns)

# Remove duplicates based on selected column
df_duplicates_removed = df.drop_duplicates(subset=column_to_remove_duplicates)

# Select columns to include in the treemap
columns = st.multiselect("Select columns for analysis", options=df_duplicates_removed.columns)

# Dictionary to store selected elements for each column
selected_elements = {}

for column in columns:
    # Select elements for each column
    elements = st.multiselect(f"Select elements for column '{column}'", options=df_duplicates_removed[column].unique())

    if len(elements) > 0:
        # Filter data based on selected elements
        filtered_data = df_duplicates_removed[df_duplicates_removed[column].isin(elements)]
        selected_elements[column] = elements
    else:
        # No elements selected, analyze the whole column
        filtered_data = df_duplicates_removed
        selected_elements[column] = None

    # Display DataFrame with selected elements
    st.write(f"Analysis for column '{column}'")
    st.write(filtered_data)

if len(columns) > 0:
    # Check if selected elements meet all criteria
    meets_all_criteria = all(selected_elements[column] is not None for column in columns)

    if meets_all_criteria:
        # Create treemap chart
        fig = px.treemap(filtered_data, path=columns)
        st.plotly_chart(fig)
    else:
        st.warning("Selected elements do not meet all criteria. Please choose elements that match all selected columns.")
else:
    st.warning("No columns selected. Please choose at least one column.")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("Histogram", "Visualization")