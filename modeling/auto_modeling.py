import streamlit as st

def run_modeling(df, task):
    from pycaret.classification import setup as clf_setup, compare_models
    from pycaret.regression import setup as reg_setup, compare_models

    target = st.selectbox("Select Target Column", df.columns)

    if len(df) < 2:
        st.error("❌ Dataset must have at least 2 rows.")
        return

    st.write("Target column class distribution:")
    st.write(df[target].value_counts())

    if task == "classification":
        if df[target].value_counts().min() < 2:
            st.error("❌ One or more classes have less than 2 examples. Please upload a more balanced dataset.")
            return
        clf_setup(df, target=target)
        best_model = compare_models()
    else:
        reg_setup(df, target=target)
        best_model = compare_models()

    st.success("✅ Modeling complete.")
    st.write(best_model)
