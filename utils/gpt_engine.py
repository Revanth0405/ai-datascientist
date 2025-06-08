import google.generativeai as genai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys
import uuid
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key="AIzaSyCKB1493rWdWRKz2z1eEO6b1V9Q3DJarrM")

def clean_generated_code(code: str) -> str:
    # Remove any markdown code block markers
    code = re.sub(r'```python\n|```\n|```', '', code)

    # Fix common string formatting issues
    code = re.sub(r'(?<!\\)"', '"', code)  # Fix unescaped quotes
    code = re.sub(r"(?<!\\)'", "'", code)  # Fix unescaped single quotes

    # Remove any data loading statements except the embedded file loading line (handled separately)
    # We will keep a placeholder for the file loading line only if GPT includes it correctly.
    # Otherwise, remove generic pd.read_csv or sns.load_dataset calls
    code = re.sub(r"df\s*=\s*pd\.read_csv\([^\n]+\)", "# 🔒 Removed: dataset loading", code)
    code = re.sub(r"sns\.load_dataset\([^\n]+\)", "# 🔒 Removed: external seaborn dataset", code)

    return code.strip()

def generate_code_with_gpt(prompt: str, df: pd.DataFrame, file_path: str):
    try:
        # Prepare a sample preview for the prompt
        sample_data = df.sample(min(5, len(df))).to_markdown(index=False)

        final_prompt = f"""
You are a helpful data science assistant.

Write executable Python code to perform:
- Exploratory Data Analysis (EDA)
- At least 5 visualizations using Matplotlib or Seaborn
- Model training for the dataset

⚠️ Important Instructions:
- The dataset must be loaded from this CSV file path:
    df = pd.read_csv(r"{file_path}")
- DO NOT include any other pd.read_csv(...) or dataset loading code.
- Always handle categorical columns (like species names or yes/no) using LabelEncoder or pd.get_dummies() if needed.
- NEVER leave a try or control block (if, else, for, while) empty; use `pass` if no action is needed.
- Only return valid Python code. No explanations or markdown formatting.

Here is a preview of the dataset:
{sample_data}
"""

        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Generate response
        response = model.generate_content(final_prompt)
        code = response.text

        # Clean and format the code
        code = clean_generated_code(code)

        # Show the code
        st.subheader("🧠 Gemini-Generated Code")
        st.code(code, language="python")

        return code

    except Exception as e:
        st.error(f"⚠️ Gemini API Error: {str(e)}")

def call_gpt_with_fix(code, error, df_sample, file_path):
    fix_prompt = f"""You are a Python expert. Fix the following code that has syntax errors.
The code should be valid Python code that can be executed directly.

Original code with error:
{code}

Error message:
{error}

Dataset must be loaded from this CSV file path exactly as:
df = pd.read_csv(r"{file_path}")

Sample data (for reference):
{df_sample.to_markdown(index=False)}

Generate only the fixed Python code without any explanations or markdown formatting.
Make sure the code is properly indented and all blocks are complete."""

    # Initialize the model
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Generate response
    response = model.generate_content(fix_prompt)
    fixed_code = response.text

    # Clean the code
    fixed_code = clean_generated_code(fixed_code)

    # Auto-fix indentation if needed
    try:
        import ast
        ast.parse(fixed_code)  # Validate syntax
    except SyntaxError:
        lines = fixed_code.split('\n')
        fixed_lines = []
        indent_level = 0
        for line in lines:
            if line.strip().startswith(('except', 'else', 'finally')):
                indent_level = max(0, indent_level - 1)
            fixed_lines.append('    ' * indent_level + line.lstrip())
            if line.strip().startswith(('try', 'if', 'for', 'while', 'def', 'class')):
                indent_level += 1
        fixed_code = '\n'.join(fixed_lines)

    return fixed_code

def auto_indent_fix(code: str) -> str:
    lines = code.split('\n')
    fixed_lines = []
    control_keywords = ('try', 'except', 'if', 'elif', 'else', 'for', 'while', 'with', 'def', 'class')

    for i, line in enumerate(lines):
        stripped = line.strip()
        fixed_lines.append(line)

        if any(re.match(rf"^{kw}\b.*:$", stripped) for kw in control_keywords):
            # Check next line exists and is not properly indented
            next_line = lines[i+1] if i+1 < len(lines) else ''
            if not next_line.strip() or not next_line.startswith((' ', '\t', '#')):
                fixed_lines.append('    pass')  # Safe fallback

    return '\n'.join(fixed_lines)

def run_gpt_code(code: str, df: pd.DataFrame, file_name: str):
    # Encode all categorical columns to numeric to avoid string to float errors
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        try:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        except Exception as e:
            st.warning(f"⚠️ Could not encode column '{col}': {e}")

    local_vars = {"df": df_encoded}

    output_folder = f"outputs/{os.path.splitext(file_name)[0]}_{str(uuid.uuid4())[:8]}"
    os.makedirs(output_folder, exist_ok=True)

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        # Fix indentation automatically before running
        cleaned_code = auto_indent_fix(code)
        exec(cleaned_code, {}, local_vars)
    except Exception as e:
        st.error(f"❌ Error running GPT-generated code:\n\n{e}")
        return
    finally:
        sys.stdout = old_stdout

    output = mystdout.getvalue()
    if output.strip():
        st.text_area("📤 Output from Code", output, height=200)

    # Show matplotlib/seaborn plots generated by the GPT code
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        image_path = os.path.join(output_folder, f"viz_{i+1}.png")
        fig.savefig(image_path)
        st.image(image_path, caption=f"Visualization {i+1}")
    plt.close("all")
