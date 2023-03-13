import streamlit as st
import matplotlib.pyplot as plt
from utils.utils import *


filename = 'dataset/housing.csv'

st.write("<h1>EDA and Prdictive Modelling Dashboard<h1>", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    'Select an option',
    ('EDA', 'Predictive Modelling'))



if option == "EDA":
    st.write('<h3>Exploratry Data Analysis and Visualization</h3>', unsafe_allow_html=True)
    st.write('Choose a plot type from the otpions below:')
    raw_data_option = st.checkbox('Show Raw data')
    if raw_data_option:
        st.dataframe(read_data(filename))

    missing_value_option = st.checkbox('Show Missing value')
    if missing_value_option:
        st.dataframe(missing_value(filename))

    datatype_option = st.checkbox('Show Data types')
    if datatype_option:
        st.dataframe(show_datatype(filename))

    descriptive_option = st.checkbox('Show Descriptive Search')
    if descriptive_option:
        st.dataframe(descriptive_search(filename))

    corr_matrix_option = st.checkbox('Show Correlation matrix')
    if corr_matrix_option:
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        dataframe = read_csv(filename, delim_whitespace=True, names=names)
        fig = pyplot.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        ticks = arange(0,14,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        st.pyplot(fig)

    histogram_option = st.checkbox('Show Histogram for each attributes')
    if histogram_option:
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        dataframe = read_csv(filename, delim_whitespace=True, names=names)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        dataframe.hist(figsize=(10,10))
        st.pyplot()

    density_option = st.checkbox('Show Density for each attributes')
    if density_option:
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        dataframe = read_csv(filename, delim_whitespace=True, names=names)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        dataframe.plot(kind='density', subplots=True, layout=(4,4), figsize=(10,10),sharex=False)
        st.pyplot()
        
    scatter_option = st.checkbox('Show Scatter plot')
    if scatter_option:
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        dataframe = read_csv(filename, delim_whitespace=True, names=names)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        scatter_matrix(dataframe, figsize=(10,10))
        st.pyplot()


else:
    st.write('<h3>Predictive Modelling</h3>', unsafe_allow_html=True)
    st.write("Choose a transform type and Model from the otpions below:")

    transform_option = st.selectbox("Select data transform", ("None", "StandardScaler", "Normalize", "MinMaxScaler") )
    model_option = st.selectbox("Select classifier", ("LogisticRegression", "SVM", "ElasticNet", "Lasso"))

    st.write(f"Here are the results of a {model_option} model:")
    result = perform_prediction(filename=filename, model_name=model_option, transform=transform_option)
    st.table(result)