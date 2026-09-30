"""
Streamlit UI for the Email/SMS Spam Detection system.

Usage:
    streamlit run src/app.py
"""

import pandas as pd
import streamlit as st

# ---------- Page configuration ----------
st.set_page_config(
    page_title="AI Email & SMS Spam Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-spam {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1.2rem;
        border-radius: 8px;
        color: #991b1b;
        margin-top: 1rem;
    }
    .result-ham {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1.2rem;
        border-radius: 8px;
        color: #166534;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Lazy-import predict & preprocessing ----------
try:
    from src.predict import predict
    from src.preprocessing import transform_text
except ImportError:
    try:
        from predict import predict
        from preprocessing import transform_text
    except ImportError:
        from predict import predict
        from data_preprocessing import transform_text


# ---------- Sidebar ----------
with st.sidebar:
    st.title("Spam Guard AI")
    st.caption("Principal Engineer AI Engine v1.0")

    st.divider()

    st.markdown("### Model Metrics")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Accuracy", "98.65%")
    col_s2.metric("Precision", "98.13%")

    col_s3, col_s4 = st.columns(2)
    col_s3.metric("Recall", "89.74%")
    col_s4.metric("F1-Score", "0.9375")

    st.divider()

    st.markdown(
        """
        **Architecture Stack**:
        - **Preprocessing**: NLTK Stemming & Stopword Filtering
        - **Vectorisation**: TF-IDF (L2-Normalised)
        - **Classifier**: Multinomial Naive Bayes (Zero-DLL NumPy Engine)
        """
    )
    st.caption("Production Ready · Ultra-Fast Inference")


# ---------- Main Content ----------
st.markdown('<div class="main-header">Email & SMS Spam Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Instantly classify messages using advanced Machine Learning & NLP analysis.</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Single Message Analysis", "Batch File Processing"])

# ================= TAB 1: SINGLE MESSAGE =================
with tab1:
    st.markdown("#### Quick Pre-loaded Samples")
    sample_cols = st.columns(4)

    sample_text = ""
    if sample_cols[0].button("Claim Prize (Spam)"):
        sample_text = "WINNER!! You have been selected to receive a $1000 cash prize! Call 09061701461 to claim NOW!"
    elif sample_cols[1].button("Meeting Invite (Ham)"):
        sample_text = "Hey John, are we still meeting for lunch at 12:30 PM tomorrow?"
    elif sample_cols[2].button("Bank Fraud Alert (Spam)"):
        sample_text = "URGENT: Your bank account has been compromised. Verify your credentials immediately at http://bit.ly/fake-bank"
    elif sample_cols[3].button("Friend Chat (Ham)"):
        sample_text = "Can you send me the python script when you get home? Thanks!"

    input_text = st.text_area(
        "Enter Message Body",
        value=sample_text,
        height=160,
        placeholder="Type or paste an email or SMS message here …",
    )

    col_action1, col_action2 = st.columns([1, 4])
    predict_clicked = col_action1.button("Analyze Message", type="primary", use_container_width=True)

    if predict_clicked or (sample_text and input_text == sample_text):
        if not input_text or not input_text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            try:
                result = predict(input_text)
                label = result["label"]
                confidence = result["confidence"]

                st.divider()

                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    if label == "spam":
                        st.markdown(
                            f"""
                            <div class="result-spam">
                                <h3>[SPAM DETECTED]</h3>
                                <p>This message exhibits patterns commonly found in unwanted or malicious communications.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-ham">
                                <h3>[SAFE (HAM)]</h3>
                                <p>This message appears legitimate and safe.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with res_col2:
                    st.markdown("##### Prediction Confidence")
                    st.progress(confidence)
                    st.metric("Confidence Score", f"{confidence:.1%}")

                # NLP Transformation Insights
                with st.expander("View NLP Preprocessing Breakdown"):
                    cleaned_tokens = transform_text(input_text)
                    st.write("**Original Text Length:**", len(input_text), "characters")
                    st.write("**Processed Tokens:**", cleaned_tokens if cleaned_tokens else "*(No alphanumeric tokens remaining)*")

            except ValueError as exc:
                st.warning(f"Warning: {exc}")
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")


# ================= TAB 2: BATCH FILE PROCESSING =================
with tab2:
    st.markdown("#### Upload CSV or TXT File for Batch Classification")
    uploaded_file = st.file_uploader("Upload CSV file containing a 'text' column", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                lines = [line.decode("utf-8").strip() for line in uploaded_file.readlines() if line.strip()]
                df_upload = pd.DataFrame({"text": lines})

            text_col = None
            for col in df_upload.columns:
                if col.lower() in ["text", "v2", "message", "email", "body"]:
                    text_col = col
                    break

            if text_col is None:
                text_col = df_upload.columns[0]

            st.info(f"Using column **'{text_col}'** for text analysis.")

            if st.button("Process Batch Messages", type="primary"):
                with st.spinner("Analyzing messages..."):
                    results_list = []
                    for txt in df_upload[text_col]:
                        try:
                            res = predict(str(txt))
                            results_list.append(res)
                        except Exception:
                            results_list.append({"label": "unknown", "confidence": 0.0})

                    df_results = df_upload.copy()
                    df_results["predicted_label"] = [r["label"] for r in results_list]
                    df_results["confidence"] = [r["confidence"] for r in results_list]

                    # Summary Metrics
                    total_count = len(df_results)
                    spam_count = int((df_results["predicted_label"] == "spam").sum())
                    ham_count = int((df_results["predicted_label"] == "ham").sum())
                    spam_pct = (spam_count / total_count * 100) if total_count > 0 else 0

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Total Processed", total_count)
                    m_col2.metric("Spam Messages", spam_count)
                    m_col3.metric("Safe (Ham) Messages", ham_count)
                    m_col4.metric("Spam Percentage", f"{spam_pct:.1f}%")

                    st.dataframe(df_results, use_container_width=True)

                    # Download button
                    csv_data = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Classification Results (CSV)",
                        data=csv_data,
                        file_name="spam_classification_results.csv",
                        mime="text/csv",
                    )

        except Exception as exc:
            st.error(f"Failed to process uploaded file: {exc}")
