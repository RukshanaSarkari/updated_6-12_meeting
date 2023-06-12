import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from kernel import app_utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

st.title("Visualization")
st.write("---")

if "loaded_df" in st.session_state:

    df = st.session_state["loaded_df"]

    st.title('Data Visualizer')

    # Select subset of columns
    cols = st.multiselect("Columns", df.columns.tolist())
    # Show dataframe
    if cols:
        st.dataframe(df[cols])
    else:
        st.dataframe(df)

    # Choose between numerical or categorical column filtering
    col_type = st.radio('Choose column type for filtering', ['Numerical', 'Categorical'])

    if col_type == 'Numerical':
        # Select column to filter
        filter_col = st.selectbox("Filter column", df.select_dtypes(include=['float64', 'int64']).columns.tolist())
        # Select range for filtering
        min_val, max_val = st.slider("Filter values", float(df[filter_col].min()), float(df[filter_col].max()),
                                     value=(float(df[filter_col].min()), float(df[filter_col].max())))
        # Filter dataframe
        filtered_df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]

    elif col_type == 'Categorical':
        filter_col = st.selectbox("Filter column", df.select_dtypes(include=['object']).columns.tolist())
        filter_val = st.multiselect("Filter values", df[filter_col].unique().tolist())
        filtered_df = df[df[filter_col].isin(filter_val)]

    st.subheader('Filtered Data')
    st.dataframe(filtered_df)

    # Select visualization type
    viz = st.selectbox("Visualization type", ['Histogram', 'Box plot', 'Bar chart', 'Heatmap'])

    if viz == 'Histogram':
        num_col = st.selectbox("Select column",
                               filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist())
        fig, ax = plt.subplots()
        ax.hist(filtered_df[num_col], bins=20)
        st.pyplot(fig)

    elif viz == 'Box plot':
        num_col = st.selectbox("Select column",
                               filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist())
        fig, ax = plt.subplots()
        ax.boxplot(filtered_df[num_col])
        st.pyplot(fig)

    elif viz == 'Bar chart':
        cat_col = st.selectbox("Select column", filtered_df.select_dtypes(include=['object']).columns.tolist())
        fig, ax = plt.subplots()
        filtered_df[cat_col].value_counts().plot(kind='bar')
        st.pyplot(fig)

    elif viz == 'Heatmap':
        if len(cols) > 2:
            fig, ax = plt.subplots()
            sns.heatmap(filtered_df[cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Please select more columns for a meaningful heatmap.")

    # df_analysis = st.session_state["df_filtered"]
    #
    # st.write(st.session_state['file_name'])
    #
    # if "Airbase" in st.session_state["file_name"]:
    #     df_analysis.columns = ["LAT", "LON"]
    #     st.map(df_analysis)
    #
    # # df_visual = pd.DataFrame(df_analysis)
    # df_visual = df_analysis.copy()
    # cols = pd.read_csv('df/metadata/column_type_desc.csv')
    # st.write(cols)
    # categorical, numerical, obj = utils.getColumnTypes(cols)
    # st.write("Here")
    # st.write(numerical)
    # cat_groups = {}
    # unique_category_val = {}
    #
    # for i in range(len(categorical)):
    #     if categorical[i] not in unique_category_val:
    #         unique_category_val[categorical[i]] = []
    #
    #     unique_category_val[categorical[i]] = utils.map_unique(df_analysis, categorical[i])
    #     # unique_category_val = {categorical[i]: utils.map_unique(df_analysis, categorical[i])}
    #
    #     if categorical[i] not in cat_groups:
    #         cat_groups[categorical[i]] = []
    #
    #     cat_groups[categorical[i]] = df_visual.groupby(categorical[i])
    #     # cat_groups = {categorical[i]: df_visual.groupby(categorical[i])}
    #
    # category = st.selectbox("Select Category ", categorical + obj)
    #
    # sizes = (df_visual[category].value_counts() / df_visual[category].count())
    #
    # labels = sizes.keys()
    #
    # max_index = np.argmax(np.array(sizes))
    # explode = [0] * len(labels)
    # explode[int(max_index)] = 0.1
    # explode = tuple(explode)
    #
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=0)
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # ax1.set_title('Distribution for categorical Column - ' + str(category))
    # st.pyplot(fig1)
    #
    # corr = df_analysis.corr(method='pearson')
    #
    # fig2, ax2 = plt.subplots()
    # # mask = np.zeros_like(corr, dtype=np.bool)
    # mask = np.full(corr.shape, True, dtype=bool)
    #
    # mask[np.triu_indices_from(mask)] = True
    # # Colors
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)
    # sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0, ax=ax2)
    # ax2.set_title("Correlation Matrix")
    # st.pyplot(fig2)
    #
    # category_object = st.selectbox("Select " + str(category), unique_category_val[category])
    # st.write(cat_groups[category].get_group(category_object).describe())
    # col_name = st.selectbox("Select Column ", numerical)
    #
    # try:
    #     st.bar_chart(cat_groups[category].get_group(category_object)[col_name])
    # except KeyError:
    #     st.warning("Something went wrong")

else:
    st.warning("Empty df table.")

# ----------------------------------------------------------------
# Page Navigation
au.display_page_navigation("TreeMap", "Data Aggregation")
