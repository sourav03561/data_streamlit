import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set up a file uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "pdf"])
# Check if a file was uploaded
if uploaded_file is not None:
    # Read the file    
    # For CSV or text files, you can read the contents
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.write(df)
        st.write(df.info())
        st.write(df.describe())
        st.write(df.describe(include=['object', 'category']))
        st.subheader('Correlation Matrix Heatmap')
        # Filter to include only numerical columns
        numerical_df = df.select_dtypes(include=['number'])

        # Correlation matrix
        corr_matrix = numerical_df.corr()
        st.write(corr_matrix)

        # Heatmap of the correlation matrix
        st.subheader('Correlation Matrix Heatmap')
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        numerical_df = df.select_dtypes(include=['number'])
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_df)

        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        explained_variance_ratio = pca.explained_variance_ratio_

        st.subheader('Principal Components')
        num_components = st.slider('Select number of principal components to view:', 1, len(explained_variance_ratio), 2)
        principal_components_df = pd.DataFrame(pca_result[:, :num_components], columns=[f'PC{i+1}' for i in range(num_components)])
        st.write(principal_components_df.head())

        st.subheader('Select Columns for Scatter Plot')
        x_column = st.selectbox('Select the X-axis variable:', df.columns)
        y_column = st.selectbox('Select the Y-axis variable:', df.columns)

        # Generate scatter plot
    if st.button('Generate Scatter Plot'):
        # Check if selected columns are numeric
        if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
            fig, ax = plt.subplots()
            ax.scatter(df[x_column], df[y_column], color='blue', alpha=0.7)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f'Scatter Plot of {x_column} vs {y_column}')
            st.pyplot(fig)
        else:
            st.error('Both selected columns must be numeric for scatter plot.')
