import google.generativeai as genai
import pandas as pd

genai.configure(api_key="AIzaSyCKB1493rWdWRKz2z1eEO6b1V9Q3DJarrM")  # replace if not done globally

def detect_task_type(df: pd.DataFrame) -> str:
    target = df.columns[-1]
    sample = df.sample(min(20, len(df)))  # Smaller sample for Gemini
    sample_md = sample.to_markdown(index=False)

    prompt = f"""
You are a machine learning expert.

Given this sample dataset and its target column **{target}**, determine if the task is a **classification** or **regression** problem.

Here is the sample data:
{sample_md}

Rules:
- If the target is numeric and represents continuous values like price, score, or measurements, it's regression.
- If the target is categorical (like labels, categories, classes), it's classification.

Only respond with **"classification"** or **"regression"**. No explanation.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    decision = response.text.strip().lower()

    if decision not in ["classification", "regression"]:
        raise ValueError(f"Invalid task type from Gemini: {decision}")

    return decision
