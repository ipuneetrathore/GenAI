import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import concurrent.futures
import datetime
import json
import plotly.graph_objects as go

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="Model Explainability Chat", layout="wide")

# Load .env from the same directory as this script (useful when running from project root)
load_dotenv()

# Read and sanitize OPENAI_API_KEY (strip surrounding quotes if present)
_raw_key = os.getenv("OPENAI_API_KEY", "")
_OPENAI_API_KEY = _raw_key.strip().strip('"').strip("'") if _raw_key else ""
client = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else None

# Debug indicator (visible in the app) if the key was loaded
_KEY_LOADED = bool(_OPENAI_API_KEY)

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


def build_datapoint_prompt(model_df, shap_df, idx, hist_df=None, top_k=2):
    """Create a prompt that describes a single datapoint using its feature values and SHAP values.
    If hist_df is provided, find top_k similar historic rows (by numeric features) and include their dispositions.
    """
    row = model_df.loc[idx]
    shap_row = shap_df.loc[idx] if idx in shap_df.index else None

    prompt = """
You are a senior data scientist. Summarize and explain why this single datapoint is anomalous.
Use the provided feature values, SHAP values (feature importances for this row), and, if available, similar historic disposition examples.
Provide: (1) short summary of the datapoint, (2) most important features causing anomaly, (3) similarity to historic dispositions and whether those were false positives or true positives (if available), and (4) one-sentence recommended next step to investigate.
"""

    prompt += "\n-- Feature values for the datapoint (index={}) --\n".format(idx)
    for c, v in row.items():
        prompt += f"- {c}: {v}\n"

    if shap_row is not None:
        prompt += "\n-- SHAP values for the datapoint (top features only) --\n"
        # show fewer top features to keep prompt small (reduce output size)
        try:
            shap_series = shap_row.abs().sort_values(ascending=False)
            top = shap_series.head(4).index.tolist()
            for f in top:
                prompt += f"- {f}: SHAP={shap_row[f]}\n"
        except Exception:
            # fallback: only show up to 6 features
            cnt = 0
            for c, v in shap_row.items():
                if cnt >= 6:
                    break
                prompt += f"- {c}: SHAP={v}\n"
                cnt += 1

    # similarity to historic dispositions
    if hist_df is not None and len(hist_df) > 0:
        prompt += "\n-- Similar historic disposition examples --\n"
        # choose numeric columns common to both
        num_cols = [c for c in model_df.select_dtypes(include=[np.number]).columns if c in hist_df.columns]
        if len(num_cols) >= 1:
            # build numeric matrices
            try:
                target = np.array(row[num_cols], dtype=float).reshape(1, -1)
                hist_vals = hist_df[num_cols].astype(float).to_numpy()
                # compute euclidean distances
                dists = np.linalg.norm(hist_vals - target, axis=1)
                idxs = np.argsort(dists)[:top_k]
                for i in idxs:
                    hist_row = hist_df.iloc[i]
                    disp = None
                    for col in ("disposition", "label", "outcome", "true_label"):
                        if col in hist_row.index:
                            disp = hist_row[col]
                            break
                    prompt += f"- distance={float(dists[i]):.4f}, disposition={disp}, row_index={i}\n"
            except Exception:
                prompt += "(could not compute numeric similarity due to non-numeric values)\n"
        else:
            prompt += "(no numeric columns in common to compute similarity)\n"

    # Instruct LLM to be concise (reduce reply length)
    try:
        max_words = int(os.getenv("EXPLAIN_MAX_WORDS", "150"))
    except Exception:
        max_words = 150

    prompt += f"\n\n-- Reply instructions: Keep the explanation concise. Limit to around {max_words} words or 5 short bullets. --\n"

    return prompt


def ask_openai(system_prompt, user_question):
    if client is None:
        return "OPENAI_API_KEY is not set. Set OPENAI_API_KEY in your .env or environment and restart the app."
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    # Timeout for OpenAI requests (seconds). Default 15s; configurable via OPENAI_TIMEOUT
    try:
        timeout_secs = int(os.getenv("OPENAI_TIMEOUT", "15"))
    except Exception:
        timeout_secs = 15

    def _call_api():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.2)),
        )

    # prepare a request log entry
    start_ts = datetime.datetime.utcnow()
    # serialize messages for logging/display
    try:
        messages_text = json.dumps(messages, indent=2, default=str)
    except Exception:
        messages_text = str(messages)

    try:
        st.session_state.setdefault("llm_log", []).append({
            "time": start_ts.isoformat(),
            "type": "request",
            "model": model,
            "prompt": system_prompt,
            "question": user_question,
            "messages": messages,
            "messages_text": messages_text,
            "status": "sending",
            "actions": ["send_request"],
        })
    except Exception:
        pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call_api)
            response = fut.result(timeout=timeout_secs)
    except concurrent.futures.TimeoutError:
        reply_text = f"OpenAI request timed out after {timeout_secs} seconds"
        # log the call
        try:
            st.session_state.setdefault("llm_log", []).append({
                "time": datetime.datetime.utcnow().isoformat(),
                "prompt": system_prompt,
                "response": reply_text,
            })
        except Exception:
            pass
        return reply_text
    except Exception as e:
        reply_text = f"OpenAI API error: {e}"
        try:
            st.session_state.setdefault("llm_log", []).append({
                "time": datetime.datetime.utcnow().isoformat(),
                "prompt": system_prompt,
                "response": reply_text,
            })
        except Exception:
            pass
        return reply_text

    try:
        reply_text = response.choices[0].message.content
    except Exception:
        try:
            reply_text = response.choices[0].text
        except Exception:
            reply_text = str(response)

    # record response log entry with actions and duration
    try:
        st.session_state.setdefault("llm_log", []).append({
            "time": datetime.datetime.utcnow().isoformat(),
            "type": "response",
            "model": model,
            "prompt": system_prompt,
            "response": reply_text,
            "messages": messages,
            "messages_text": messages_text,
            "status": "ok",
            "actions": ["send_request", "receive_response", "parse_response"],
            "duration_seconds": (datetime.datetime.utcnow() - start_ts).total_seconds(),
        })
        # trim to last 200
        log = st.session_state.get("llm_log", [])
        if len(log) > 200:
            st.session_state["llm_log"] = log[-200:]
    except Exception:
        pass

    return reply_text


# -----------------------
# UI
# -----------------------
st.title("📊 Anomaly Analysis Assistant")

# Warn if OpenAI key not loaded
if not _KEY_LOADED:
    st.warning("OPENAI_API_KEY not found. Add it to V1/.env or your environment to enable explanations.")

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
# Load Data and UI
# -----------------------
if model_file and shap_file:
    model_df = pd.read_csv(model_file)
    shap_df = pd.read_csv(shap_file)
    hist_df = pd.read_csv(hist_file) if hist_file else None

    st.success("✅ Required files uploaded successfully")

    # Layout: main area + right-side background info
    main_col, side_col = st.columns([3, 1])

    with main_col:
        st.subheader("Model Output — Anomalies")
        if "is_anomaly" in model_df.columns:
            anomalies_df = model_df[model_df["is_anomaly"] == 1]
            if anomalies_df.empty:
                st.info("No rows with is_anomaly==1 found in the uploaded model output.")
                st.dataframe(model_df.head())
            else:
                st.dataframe(anomalies_df.head(500))

                # --- Dataset summary below the anomalies table ---
                try:
                    total_records = len(model_df)
                    total_anomalies = len(anomalies_df)

                    def _is_alert_val(v):
                        if pd.isna(v):
                            return False
                        s = str(v).strip().lower()
                        if s in ("1", "true", "yes", "tp", "t", "y"):
                            return True
                        if "alert" in s or "true positive" in s or "fraud" in s:
                            return True
                        return False

                    # Prefer analyst_commentary from historic dispositions if available
                    disposition_alert_count = 0
                    matching_count = 0

                    if hist_df is not None and "analyst_commentary" in hist_df.columns:
                        # count disposition-like entries in historic analyst commentary
                        disposition_alert_count = int(hist_df["analyst_commentary"].apply(_is_alert_val).sum())
                        # (not computing global top trends per user request)

                        # Try to robustly match anomalies to historic rows using likely key columns
                        matching_count = 0
                        matched_top_patterns = {}
                        try:
                            # find likely key columns shared between the two dataframes
                            key_indicators = ("id", "case", "request", "uid", "key", "txn", "transaction")
                            shared_cols = [c for c in model_df.columns if c in hist_df.columns]
                            key_cols = [c for c in shared_cols if any(k in c.lower() for k in key_indicators)]

                            if key_cols:
                                merged = anomalies_df.merge(hist_df, on=key_cols, how="inner", suffixes=("_m","_h"))
                                if not merged.empty and "analyst_commentary" in merged.columns:
                                    matching_count = int(merged["analyst_commentary"].apply(_is_alert_val).sum())
                                    matched_top_patterns = merged["analyst_commentary"].value_counts().head(5).to_dict()
                            else:
                                # no obvious key columns: try exact-match on all shared non-numeric string columns (heuristic)
                                shared_string_cols = [c for c in shared_cols if model_df[c].dtype == object and hist_df[c].dtype == object]
                                if shared_string_cols:
                                    merged = anomalies_df.merge(hist_df, on=shared_string_cols, how="inner")
                                    if not merged.empty and "analyst_commentary" in merged.columns:
                                        matching_count = int(merged["analyst_commentary"].apply(_is_alert_val).sum())
                                        matched_top_patterns = merged["analyst_commentary"].value_counts().head(5).to_dict()
                                else:
                                    matching_count = 0
                        except Exception:
                            matching_count = 0

                    else:
                        # fallback: detect disposition-like column in model_df
                        disp_candidates = [c for c in model_df.columns if any(k in c.lower() for k in ("disposition", "label", "outcome", "status", "comment"))]
                        disp_col = disp_candidates[0] if disp_candidates else None
                        if disp_col:
                            disposition_alert_count = int(model_df[disp_col].apply(_is_alert_val).sum())
                            matching_count = int(anomalies_df[disp_col].apply(_is_alert_val).sum())
                            # (not computing top_trends here)
                        else:
                            # compute top numeric-difference trends as a last resort
                            num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
                            if len(num_cols) > 0:
                                diffs = (anomalies_df[num_cols].mean() - model_df[num_cols].mean()).abs().sort_values(ascending=False)
                                # (not computing top_trends here)

                    st.markdown("**Quick summary**")
                    st.write(f"- **Total records:** {total_records}")
                    st.write(f"- **Total anomalies (is_anomaly==1):** {total_anomalies}")
                    st.write(f"- **Total disposition alert count**: {disposition_alert_count}")
                    st.write(f"- **Model alerts matching disposition**: {matching_count}")
                    if 'matched_top_patterns' in locals() and matched_top_patterns:
                        st.write("- **Top matching commentary phrases (from matched historic rows):**")
                        st.json(matched_top_patterns)
                except Exception as e:
                    st.warning(f"Could not compute summary: {e}")
        else:
            st.warning("Column 'is_anomaly' not found; showing first rows of the full dataset instead.")
            st.dataframe(model_df.head())

        st.divider()
        st.subheader("💬 Inspect & Explain Single Anomalous Datapoint")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Create human-friendly labels for selection: use index and first few columns
        def _label_for_idx(i):
            sample_cols = list(model_df.columns[:3])
            vals = ", ".join([f"{c}={model_df.loc[i,c]}" for c in sample_cols])
            return f"idx={i}: {vals}"

        # Limit selectable datapoints to anomalies where possible
        if "is_anomaly" in model_df.columns:
            anomalies_df = model_df[model_df["is_anomaly"] == 1]
            if not anomalies_df.empty:
                selection_indices = list(anomalies_df.index)
            else:
                selection_indices = list(model_df.index)
        else:
            selection_indices = list(model_df.index)

        options = [_label_for_idx(i) for i in selection_indices]
        sel_label = st.selectbox("Select a datapoint to explain", options)
        selected_idx = int(sel_label.split(":")[0].split("=")[1])

        # show selected row (features only)
        st.markdown("**Selected datapoint (features)**")
        st.write(model_df.loc[[selected_idx]])

        # Plot: compare selected datapoint numeric feature values to dataset 99th percentile
        try:
            num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                pct99 = model_df[num_cols].quantile(0.99)
                selected_vals = model_df.loc[selected_idx, num_cols]

                # prepare x (feature names) and y values
                x = list(num_cols)
                y_pct = [float(pct99[c]) if not pd.isna(pct99[c]) else None for c in x]
                y_sel = [float(selected_vals[c]) if not pd.isna(selected_vals[c]) else None for c in x]

                fig = go.Figure()
                # swap axes: features on y-axis, values on x-axis
                fig.add_trace(go.Scatter(x=y_pct, y=x, mode='lines+markers', name='99th percentile — overall features distribution', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=y_sel, y=x, mode='lines+markers', name='Alerted event — selected alert', line=dict(color='red')))
                fig.update_layout(title='Feature comparison: Dataset 99th percentile vs selected alert', xaxis_title='Value', yaxis_title='Feature', legend_title='Series')

                st.plotly_chart(fig, use_container_width=True)
                st.caption('Blue line: 99th percentile of each numeric feature across the full model output dataset. Red line: numeric values for the selected alerted datapoint. (Features on the Y axis, values on the X axis)')
        except Exception:
            pass

        # Chat input is optional; default question created when button pressed
        user_input = st.chat_input("Ask a question about the selected datapoint (or click 'Explain datapoint')...")

        if st.button("Explain datapoint") or (user_input and user_input.strip()):
            question = user_input if user_input and user_input.strip() else "Please summarize and explain this anomalous datapoint."
            st.session_state.chat_history.append({"role":"user","content":question})

            system_prompt = build_datapoint_prompt(model_df, shap_df, selected_idx, hist_df=hist_df)
            with st.spinner("Generating explanation..."):
                answer = ask_openai(system_prompt, question)

            st.session_state.chat_history.append({"role":"assistant","content":answer})

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"]) 

    # Right column: What's happening in the background
    with side_col:
        st.markdown("**What's happening in the background**")
        llm_log = st.session_state.get("llm_log", [])
        if not llm_log:
            st.info("No LLM calls yet.")
        else:
            # show most recent entries (up to 200)
            for entry in reversed(llm_log[-200:]):
                ts = entry.get("time")
                etype = entry.get("type", "entry")
                model = entry.get("model", "")
                status = entry.get("status", "")
                actions = entry.get("actions", []) or []
                duration = entry.get("duration_seconds", None)
                question = entry.get("question", "")
                header = f"{ts}"
                if etype:
                    header += f" · {etype}"
                if model:
                    header += f" · {model}"

                with st.expander(header):
                    st.markdown("**Type:** " + etype)
                    if model:
                        st.markdown("**Model:** " + model)
                    if status:
                        st.markdown("**Status:** " + status)
                    if actions:
                        st.markdown("**Actions taken by LLM / agent:**")
                        for a in actions:
                            st.write(f"- {a}")
                    if duration is not None:
                        try:
                            st.markdown(f"**Duration (s):** {float(duration):.3f}")
                        except Exception:
                            st.markdown(f"**Duration (s):** {duration}")
                    if question:
                        st.markdown("**User question:**")
                        st.code(question, language="text")

                    st.markdown("**Full messages sent to LLM:**")
                    # show the JSON-serialized messages (system+user) if available
                    if entry.get("messages_text"):
                        st.code(entry.get("messages_text", ""), language="json")
                    else:
                        st.code(str(entry.get("messages", entry.get("prompt", ""))), language="text")

                    st.markdown("**Prompt sent (system prompt only):**")
                    st.code(entry.get("prompt", ""), language="text")
                    st.markdown("**Response received:**")
                    st.code(entry.get("response", ""), language="text")

else:
    st.info("⬅️ Please upload both mandatory CSV files to start chatting.")
