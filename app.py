import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="Model Explainability Chat", layout="wide")

# Load .env from the same directory as this script (useful when running from project root)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Debug indicator (visible in the app) if the key was loaded
_KEY_LOADED = bool(os.getenv("OPENAI_API_KEY"))

# -----------------------
# Helper Functions
# -----------------------
def summarize_dataframe(df, name, max_rows=5):
    """Create a compact text summary of a dataframe for LLM context"""
    summary = f"""
Dataset: {name}
Columns: {list(df.columns)}
Shape: {df.shape}

Sample rows:
{df.head(max_rows).to_csv(index=False)}
"""
    return summary


def build_system_prompt(model_df, shap_df, hist_df=None):
    prompt = """
You are a senior data scientist helping a user analyze ML model outputs and SHAP values.
Answer questions using ONLY the data provided.
If a question cannot be answered from the data, say so clearly.
Explain insights in simple, business-friendly language.
"""

    prompt += summarize_dataframe(model_df, "Model Output")
    prompt += summarize_dataframe(shap_df, "SHAP Values")

    if hist_df is not None:
        prompt += summarize_dataframe(hist_df, "Historic Disposition Patterns")

    return prompt


def ask_openai(system_prompt, user_question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# -----------------------
# UI
# -----------------------
st.title("📊 Model Output & SHAP Chat Assistant")

st.sidebar.header("📁 Upload CSV Files")

model_file = st.sidebar.file_uploader(
    "Upload Model Output CSV (mandatory)",
    type="csv"
)

shap_file = st.sidebar.file_uploader(
    "Upload SHAP Values CSV (mandatory)",
    type="csv"
)

hist_file = st.sidebar.file_uploader(
    "Upload Historic Disposition CSV (optional)",
    type="csv"
)

# -----------------------
# Load Data
# -----------------------
if model_file and shap_file:
    model_df = pd.read_csv(model_file)
    shap_df = pd.read_csv(shap_file)
    hist_df = pd.read_csv(hist_file) if hist_file else None

    st.success("✅ Required files uploaded successfully")

    with st.expander("🔍 Preview Uploaded Data"):
        st.subheader("Model Output")
        st.dataframe(model_df.head())

        st.subheader("SHAP Values")
        st.dataframe(shap_df.head())

        if hist_df is not None:
            st.subheader("Historic Disposition Patterns")
            st.dataframe(hist_df.head())

    # -----------------------
    # Chat Interface
    # -----------------------
    st.divider()
    st.subheader("💬 Ask Questions About the Data")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question about the uploaded data...")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        system_prompt = build_system_prompt(model_df, shap_df, hist_df)

        with st.spinner("Thinking..."):
            answer = ask_openai(system_prompt, user_input)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

else:
    st.info("⬅️ Please upload both mandatory CSV files to start chatting.")
