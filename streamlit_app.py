from pathlib import Path
from uuid import uuid4
from datetime import date, time

import numpy as np
import pandas as pd
import streamlit as st

from app.predictor import TransactionScorer

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Personal Finance Transaction Intelligence",
    page_icon="💳",
    layout="wide",
)

st.title("Personal Finance Transaction Intelligence")
st.write("Automatically categorize spending and flag unusual transactions.")
st.caption(
    "Academic demonstration using models trained on synthetic data. "
    "An alert means review is needed; it does not confirm fraud."
)


@st.cache_resource
def load_scorer():
    return TransactionScorer(ROOT)


try:
    scorer = load_scorer()
except Exception as exc:
    st.error(f"Could not load the models: {exc}")
    st.stop()


def score_transactions(frame):
    required = [
        "transaction_id", "user_id", "transaction_date",
        "description", "merchant", "amount",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    frame = frame[required].copy()

    if frame.empty:
        raise ValueError("Provide at least one transaction.")
    if frame[required].isna().any().any():
        raise ValueError("Required fields cannot contain missing values.")

    for column in ["transaction_id", "user_id", "description", "merchant"]:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{column} cannot be blank.")

    frame["transaction_date"] = pd.to_datetime(
        frame["transaction_date"], errors="raise"
    )
    if frame["transaction_date"].dt.tz is not None:
        raise ValueError("Use dates without timezone offsets.")

    frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
    if not np.isfinite(frame["amount"]).all() or frame["amount"].le(0).any():
        raise ValueError("Amounts must be positive, finite numbers.")

    # Demonstrate scoring after the model's stored transaction history.
    if frame["transaction_date"].min() <= scorer.raw_history[
        "transaction_date"
    ].max():
        raise ValueError("Use transaction dates on or after July 1, 2026.")

    return scorer.score(frame, update_history=False)


def show_results(results):
    st.subheader("Results")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.caption(
        "Confidence describes the category prediction. "
        "The anomaly score is a relative unusualness score, "
        "not a probability of fraud."
    )
    st.download_button(
        "Download results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="transaction_predictions.csv",
        mime="text/csv",
    )


single_tab, batch_tab, model_tab = st.tabs(
    ["Single transaction", "CSV upload", "About the models"]
)

with single_tab:
    with st.form("transaction_form"):
        user = st.selectbox(
            "User profile",
            sorted(scorer.raw_history["user_id"].unique()),
        )
        merchant = st.text_input("Merchant", value="Kroger")
        description = st.text_input(
            "Statement description", value="POS PURCHASE KROGER #4521"
        )
        amount = st.number_input(
            "Amount ($)", min_value=0.01, value=74.30, step=1.00
        )
        transaction_day = st.date_input(
            "Transaction date",
            value=date(2026, 7, 2),
            min_value=date(2026, 7, 1),
        )
        transaction_time = st.time_input(
            "Transaction time", value=time(18, 10)
        )
        submitted = st.form_submit_button("Analyze transaction")

    if submitted:
        frame = pd.DataFrame([{
            "transaction_id": f"WEB-{uuid4().hex}",
            "user_id": user,
            "transaction_date": (
                f"{transaction_day} {transaction_time}"
            ),
            "description": description,
            "merchant": merchant,
            "amount": amount,
        }])
        try:
            with st.spinner("Analyzing transaction..."):
                results = score_transactions(frame)
            show_results(results)
        except Exception as exc:
            st.error(str(exc))

with batch_tab:
    st.write(
        "Upload a CSV containing transaction_id, user_id, "
        "transaction_date, description, merchant, and amount."
    )
    st.caption(
        "Use unique transaction IDs and dates on or after July 1, 2026. "
        "Use the existing user profiles for history-based comparisons."
    )

    sample = pd.DataFrame([{
        "transaction_id": "UPLOAD-001",
        "user_id": "U003",
        "transaction_date": "2026-07-02 18:10:00",
        "description": "POS PURCHASE KROGER #4521",
        "merchant": "Kroger",
        "amount": 74.30,
    }])
    st.download_button(
        "Download example CSV",
        sample.to_csv(index=False).encode("utf-8"),
        "example_transactions.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None and st.button("Analyze CSV"):
        try:
            frame = pd.read_csv(
                uploaded,
                dtype={"transaction_id": str, "user_id": str},
            )
            with st.spinner("Analyzing transactions..."):
                results = score_transactions(frame)
            show_results(results)
        except Exception as exc:
            st.error(str(exc))

with model_tab:
    st.write(
        "Category prediction combines a text model with a behavioral "
        "model. Anomaly detection combines Isolation Forest and "
        "One-Class SVM scores."
    )
    st.json(scorer.active)
    st.write(
        "The models use stored synthetic transaction history. "
        "Submitted transactions do not update that history."
    )