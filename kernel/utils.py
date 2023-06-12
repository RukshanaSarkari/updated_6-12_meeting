import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor

# from pandas_profiling import ProfileReport

def isCategorical(col):
    unis = np.unique(col)
    if len(unis) < 0.2 * len(col):
        return True
    return False


# def getProfile(data):
#     report = ProfileReport(data)
#     report.to_file(output_file = 'data/output.html')

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
