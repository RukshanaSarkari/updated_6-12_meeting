import streamlit as st
import pandas as pd
from kernel import data_utils as du
from streamlit_extras.switch_page_button import switch_page
from kernel.persist import load_widget_state
from kernel import app_utils as au
import shutil

load_widget_state()

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
# ----------------------------------------------------------------

with st.sidebar:
    au.display_csv_collection()


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# If the session state variables were lost (the page was refreshed?), navigate the user back to the homepage
if "data_dir" not in st.session_state:
    switch_page("Homepage")

st.title("Data")
st.write("---")

# ----------------------------------------------------------------
# Import Data Section

st.header("Import Data")

uploaded_files = st.file_uploader("Option 1: Select one or more .csv to import:", accept_multiple_files=True)
local_data_folder = st.text_input(
    "Option 2: Specify a directory of the csv files to import:",
    help="Provide the absolute path to this directory"
)

if uploaded_files:
    # Save uploaded csv files to the app df directory
    for file_obj in uploaded_files:
        new_csv = pd.read_csv(file_obj)
        new_csv.to_csv(f"{st.session_state['data_dir']}//{file_obj.name}", index=False)

elif local_data_folder:
    # List all .csv files in the folder
    data_files = du.get_files_in_folder(local_data_folder, "csv")
    # Save a copy of these csv files to the app df directory
    for file in data_files:
        shutil.copy(file, f"{st.session_state['data_dir']}//{file}")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("Homepage", "Table View")
