import json
import joblib

import pandas as pd
import streamlit as st

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from kernel.persist import load_widget_state
from kernel import app_utils as au

# Custom classes
from kernel import app_utils
import os

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

"""This application helps in running machine learning models without having to write explicit code 
by the user. It runs some basic models and let's the user select the X and y variables. 
"""

if "loaded_df" in st.session_state:

    df = st.session_state["loaded_df"]

    # Create the model parameters dictionary
    params = {}

    # Use two column technique
    col1, col2 = st.columns(2)

    # Design column 1
    y_var = col1.radio("Select the variable to be predicted (y)", options=df.columns)

    # Design column 2
    x_var = col2.multiselect("Select the variables to be used for prediction (X)", options=df.columns)

    # Check if len of x is not zero
    if len(x_var) == 0:
        st.error("You have to put in some X variable and it cannot be left empty.")

    # Check if y not in X
    if y_var in x_var:
        st.error("Warning! Y variable cannot be present in your X-variable.")

    # Option to select prediction type
    pred_type = st.radio("Select the type of process you want to run.",
                         options=["Regression", "Classification"],
                         help="Write about reg and classification")

    # Add to model parameters
    params = {
        'X': x_var,
        'y': y_var,
        'pred_type': pred_type,
    }

    # if st.button("Run Models"):

    st.write(f"**Variable to be predicted:** {y_var}")
    st.write(f"**Variable to be used for prediction:** {x_var}")

    try:
        # Divide the df into test and train set
        X = df[x_var]
        y = df[y_var]

        # Perform df imputation
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")

        # Perform encoding
        X = pd.get_dummies(X)

        # Check if y needs to be encoded
        if not utils.isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Print all the classes
            st.write("The classes and the class allotted to them is the following:-")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")

        # Perform train test splits
        st.markdown("#### Train Test Splitting")
        size = st.slider("Percentage of value division",
                         min_value=0.1,
                         max_value=0.9,
                         step=0.1,
                         value=0.8,
                         help="This is the value which will be used to divide the df for training and testing. "
                              "Default = 80%")

        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        st.write("Number of training samples:", x_train.shape[0])
        st.write("Number of testing samples:", x_test.shape[0])

        # Save the model params as a json file
        with open('df/metadata/model_params.json', 'w') as json_file:
            json.dump(params, json_file)

        ''' RUNNING THE MACHINE LEARNING MODELS '''
        if pred_type == "Regression":
            st.write("Running Regression Models on Sample")

            # Table to store model and accuracy
            model_r2 = []

            # Linear regression model
            lr_model = LinearRegression()
            lr_model.fit(x_train, y_train)
            lr_r2 = lr_model.score(x_test, y_test)
            model_r2.append(['Linear Regression', lr_r2])

            # Decision Tree model
            dt_model = DecisionTreeRegressor()
            dt_model.fit(x_train, y_train)
            dt_r2 = dt_model.score(x_test, y_test)
            model_r2.append(['Decision Tree Regression', dt_r2])

            # Save one of the models
            if dt_r2 > lr_r2:
                # save decision tree
                joblib.dump(dt_model, 'df/metadata/model_reg.sav')
            else:
                joblib.dump(lr_model, 'df/metadata/model_reg.sav')

            # Make a dataframe of results
            results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
            st.dataframe(results)

        if pred_type == "Classification":
            st.write("Running Classification Models on Sample")

            # Table to store model and accuracy
            model_acc = []

            # Logistic regression model
            lc_model = LogisticRegression()
            lc_model.fit(x_train, y_train)
            lc_acc = lc_model.score(x_test, y_test)
            model_acc.append(['Logistic Regression', lc_acc])

            # Decision Tree model
            dtc_model = DecisionTreeClassifier()
            dtc_model.fit(x_train, y_train)
            dtc_acc = dtc_model.score(x_test, y_test)
            model_acc.append(['Decision Tree Regression', dtc_acc])

            # Save one of the models
            if dtc_acc > lc_acc:
                # save decision tree
                joblib.dump(dtc_model, 'df/metadata/model_classification.sav')
            else:
                joblib.dump(lc_model, 'df/metadata/model_classification.sav')

            # Make a dataframe of results
            results = pd.DataFrame(model_acc, columns=['Models', 'Accuracy']).sort_values(by='Accuracy',
                                                                                          ascending=False)
            st.dataframe(results)
    except:
        st.warning("Something needs to be fixed here.")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("Data Aggregation", "")
