import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from kernel.persist import load_widget_state
from kernel import data_utils as du
from kernel import app_utils as au

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# App configuration and initialization
# ----------------------------------------------------------------
# ----------------------------------------------------------------

st.set_page_config(
    page_title="AD-ARM Visualization",
    page_icon=":globe_with_meridians:",
    layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        'About': "# AD-ARM Visualization Tool."
    }
)

load_widget_state()

# Import all .csv files in the folder
csv_files = du.get_files_in_folder("data//imported_csv_files", "csv")

# Import metadata files
if csv_files:
    n_csv = len(csv_files)
    meta_files = du.get_files_in_folder("data//metadata", "csv")

    if "csv_metadata.csv" in meta_files:
        csv_meta_df = du.load_csv("data//metadata//csv_metadata.csv")

    else:
        # Generate metadata for each of the csv files
        csv_meta_df = du.generate_csv_metadata(csv_files)

    csv_dfs = dict()
    var_meta_dfs = dict()
    for file in csv_files:

        csv_dfs[file] = du.load_csv("data//imported_csv_files//" + file)

        if file in meta_files:
            var_meta_dfs[file] = du.load_csv("data//metadata//meta-" + file)
        else:
            var_meta_dfs[file] = \
                du.generate_var_metadata("data//imported_csv_files//", file)

# Initialize session state variables
if "page" not in st.session_state:
    # Initialize session state.
    st.session_state.update({
        # Default page
        "page": "Homepage",

        # Resources
        "data_dir": "data//imported_csv_files",
        "csv_files": csv_files,
        "csv_meta_df": csv_meta_df,
        "csv_dfs": csv_dfs,
        "var_meta_dfs": var_meta_dfs,
        "n_vars": 0,
        "df": None,
        "edit_data": False,

        # Widget Initializations
        "csv_radio": csv_files[0],

    })

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
# ----------------------------------------------------------------

with st.sidebar:
    st.write("Add a homepage sidebar if desired")

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Main Page
# ----------------------------------------------------------------
# ----------------------------------------------------------------
st.title("AD-ARM Visualization Tool")
# st.write("---")

st.write("Add a description of the app and perhaps an image")

with st.expander("Start New Session"):
    new_session_cols = st.columns([4, 1], gap="small")

    with new_session_cols[0]:
        new_session_name = st.text_input("Session Name:")

    with new_session_cols[1]:
        st.write(" ")  # Added for vertical space
        st.write(" ")  #
        new_session_save_button = st.button("SAVE")

with st.expander("Load Saved Session"):
    st.write("Add this later")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("", "Data")
