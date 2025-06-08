import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def perform_eda(df):
    st.subheader("📊 EDA Report")

    st.write("### Dataset Summary")
    st.write(df.describe())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Correlation Heatmap")
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)