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
    st.write("Data Import Sidebar")

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
        new_csv.to_csv(f"{st.session_state['data_dir']}\\{file_obj.name}", index=False)

elif local_data_folder:
    # List all .csv files in the folder
    data_files = du.get_files_in_folder(local_data_folder, "csv")
    # Save a copy of these csv files to the app df directory
    for file in data_files:
        shutil.copy(file, f"{st.session_state['data_dir']}\\{file}")

# ----------------------------------------------------------------
# CSV File Section

st.header("CSV Collection")

# List all .csv files in the folder
csv_files = du.get_files_in_folder(st.session_state['data_dir'], "csv")

if csv_files:

    # Display CSVs in a df edit table
    csv_collection_cols = st.columns([3, 1], gap="small")

    with csv_collection_cols[0]:

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
                key="cvs_picker",
                hide_index=True,
                height=37 + 35 * len(csv_files),
            )

            if "edited_rows" in st.session_state["cvs_picker"]:
                edited_rows = st.session_state["cvs_picker"]["edited_rows"]

                old_csv_names = []
                new_csv_names = []
                csv_files_selected = []
                rows_selected = []

                # Scan through the edited rows and collect all the new names
                for row, val in edited_rows.items():

                    if "CSV collection" in val:
                        old_csv_names.append(f"{st.session_state['data_dir']}\\{csv_files[row]}")
                        new_csv_names.append(f"{st.session_state['data_dir']}\\{val['CSV collection']}")

                    if "*" in val:
                        csv_files_selected.append(f"{st.session_state['data_dir']}\\{csv_files[row]}")
                        rows_selected.append(row)

    with csv_collection_cols[1]:

        # Display an update csv-name button
        save_csv_button = st.button(
            "Save",
            help="Any csv-file names will be saved."
        )

        if save_csv_button:
            du.rename_csv_files(old_csv_names, new_csv_names)

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

        # # Display an view csv button
        # view_csv_button = st.button(
        #     "View",
        #     help="Views first selected csv-file from the collection."
        # )

    st.session_state["csv_df"] = csv_df

else:
    st.warning("No CSV files imported.")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("Homepage", "Table View")