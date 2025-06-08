import streamlit as st
import pandas as pd
import os
from cleaning.data_cleaner import clean_data
from eda.auto_eda import perform_eda
from modeling.auto_modeling import run_modeling
from utils.strategy import detect_task_type
from utils.gpt_engine import generate_code_with_gpt, run_gpt_code

st.set_page_config(layout="wide")
st.title("🧠 AI Data Scientist - V2")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    # Save uploaded file to disk
    save_folder = "uploaded_data"
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load dataframe from saved file path
    df = pd.read_csv(file_path)
    st.write("### Raw Data Preview", df.head())

    # Data cleaning
    df_clean = clean_data(df)
    st.success("✅ Data cleaned successfully.")

    with st.expander("🧹 Cleaned Data Preview"):
        st.dataframe(df_clean)

    # Perform EDA
    perform_eda(df_clean)

    # Detect task type
    task = detect_task_type(df_clean)
    st.info(f"🔍 Detected Task: **{task}**")

    # Run modeling
    run_modeling(df_clean, task)

    st.subheader("🤖 GPT-Generated Code")
    with st.expander("3️⃣ GPT-Powered Dynamic Code Generation"):
        if st.button("Generate with GPT"):
            gpt_code = generate_code_with_gpt("Generate EDA code", df_clean, file_path)
    
            if gpt_code:
                st.code(gpt_code, language="python")

                # Run and display GPT-generated code output
                run_gpt_code(gpt_code, df_clean, uploaded_file.name)

                # Download GPT code as a file
                st.download_button(
                    label="📥 Download Generated Code",
                    data=gpt_code,
                    file_name="gpt_analysis.py",
                    mime="text/x-python"
                )
            else:
                st.error("❌ GPT did not return any code. Please try again.")
