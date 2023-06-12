import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor


@st.cache_data
def rename_csv_files(old_files, new_files):
    for old, new in zip(old_files, new_files):
        os.rename(old, new)


@st.cache_data
def delete_csv_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)


# Returns all files in "path" with the specified file extension "ext"
@st.cache_data
def get_files_in_folder(path=None, ext="csv"):
    if not os.path.exists(path):
        return None

    files = []
    for file in os.listdir(path):
        if file.endswith("." + ext):
            files.append(file)
    return files


# Returns all tables in a SQLite database specified by "db_path"
@st.cache_data
def get_tables_in_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    conn.close()
    return tables


@st.cache_data
def load_csv(sel_csv, path=None):
    if path:
        sel_csv = f"{path}//{sel_csv}"

    # # set and load csv selected csv
    # csv_files = st.session_state["csv_df"]["CSV collection"]
    # csv_path = f"{st.session_state['data_dir']}//{sel_csv}"
    #
    # csv_index = int(csv_files[csv_files == sel_csv].index[0])

    # Load the DataFrame from csv
    # loaded_df = pd.read_csv(sel_csv, index_col=0)
    # loaded_df.reset_index(inplace=True)

    # return loaded_df, csv_index
    return pd.read_csv(sel_csv)


def generate_var_metadata(path, csv):
    df = pd.read_csv(path + csv)

    n_vars = len(df.columns)

    data_type = [""] * n_vars
    n_vals = [0] * n_vars
    n_missing = [0] * n_vars
    n_unique = [0] * n_vars
    is_cat = [False] * n_vars
    is_num = [False] * n_vars
    is_ord = [False] * n_vars
    is_res = [False] * n_vars

    for k, name in enumerate(df.columns):
        data_type[k] = str(df[name].dtype)
        n_vals[k] = df[name].count()
        n_missing[k] = df[name].isnull().sum()
        n_unique[k] = df[name].nunique()
        is_cat[k] = n_unique[k] < 0.05 * n_vals[k]
        is_num[k] = pd.api.types.is_numeric_dtype(df[name].dtype)
        is_ord[k] = is_num[k]
        is_res[k] = \
            name != "design_point" and \
            name != "iteration" and \
            name[0].islower()

    var_df = pd.DataFrame(
        data=zip(data_type, n_vals, n_missing, n_unique, is_cat, is_num, is_ord, is_res),
        index=df.columns,
        columns=["data type", "n values", "n missing", "n unique", "categorical", "numeric", "ordinal", "response"]
    )

    var_df.to_csv("data//metadata//meta-" + csv)
    return var_df

    # st.write(df.value_counts())
    # val_cts = df.value_counts()

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


def generate_csv_metadata(csv_files):

    n_files = len(csv_files)
    name = [""] * n_files
    location = [""] * n_files
    n_rows = [0] * n_files
    size = [0] * n_files

    for k, file in enumerate(csv_files):
        name[k] = file
        location[k] = "data//imported_csv_files//" + file
        file_size_bytes = os.path.getsize(location[k])
        file_size_mb = file_size_bytes / (1024 * 1024)
        size[k] = round(file_size_mb, 2)
        df = pd.read_csv(location[k])
        n_rows[k] = df.shape[0]

    meta_df = pd.DataFrame(
        data=zip(name, location, size, n_rows),
        columns=["name", "location", "size (mb)", "n_rows"],
    )

    meta_df.to_csv("data//metadata//csv_metadata.csv")


def filter_table_columns():
    tbl = st.session_state["df"]
    cols = st.session_state["vars"]
    st.session_state["df"] = tbl.loc[:, cols]


def get_csv_info(df):
    df_types = pd.DataFrame(df.dtypes)
    df_nulls = df.count()

    df_null_count = pd.concat([df_types, df_nulls], axis=1)
    df_null_count = df_null_count.reset_index()

    # Reassign column names
    col_names = ["features", "types"]
    df_null_count.columns = col_names

    return df_null_count


def generate_meta_data(df):
    col = df.columns
    column_type = []
    categorical = []
    obj = []
    numerical = []
    for i in range(len(col)):
        if isCategorical(df[col[i]]):
            column_type.append((col[i], "categorical"))
            categorical.append(col[i])

        elif is_numeric_dtype(df[col[i]]):
            column_type.append((col[i], "numerical"))
            numerical.append(col[i])

        else:
            column_type.append((col[i], "object"))
            obj.append(col[i])

    return column_type


def is_categorical(col):
    unis = np.unique(col)
    if len(unis) < 0.2 * len(col):
        return True
    return False


# def getProfile(df):
#     report = ProfileReport(df)
#     report.to_file(output_file = 'df/output.html')

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


def isNumerical(col):
    return is_numeric_dtype(col)


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
