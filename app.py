"""
Streamlit deployment for the national housing price models.
"""

import pickle
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Housing Value Predictor", layout="centered")


@st.cache_resource
def load_artifacts():
    with open("housing_models.pkl", "rb") as f:
        return pickle.load(f)


artifacts = load_artifacts()
models = artifacts["models"]
encoder = artifacts["encoder"]
scaler = artifacts["scaler"]
numeric_cols = artifacts["numeric_cols"]
categorical_cols = artifacts["categorical_cols"]
medians = artifacts["medians"]
modes = artifacts["modes"]
feature_order = artifacts["feature_order"]
results_df = artifacts["results_df"]
categorical_options = artifacts["categorical_options"]


def build_input_row(raw_values: dict) -> pd.DataFrame:
    """Turn a dict of raw feature values into the encoded row every model expects."""
    row = pd.DataFrame([raw_values])

    for col in numeric_cols:
        row[col + "_missing"] = row[col].isna().astype("int8")
        row[col] = row[col].fillna(medians[col])

    for col in categorical_cols:
        row[col] = row[col].fillna(modes[col])

    encoded = pd.DataFrame(
        encoder.transform(row[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=row.index,
    )
    row = pd.concat([row.drop(columns=categorical_cols), encoded], axis=1)
    row = row.reindex(columns=feature_order, fill_value=0)
    return row


def predict(model_name: str, row: pd.DataFrame) -> float:
    model = models[model_name]
    if model_name == "SVR":
        row_input = scaler.transform(row)
    else:
        row_input = row
    pred_log = model.predict(row_input)[0]
    return float(np.expm1(pred_log))


# ------------------------------------------------------------------ sidebar
st.title("🏠 National Housing Value Predictor")
st.write("Fill in the details below, pick a model (or compare all three), and get a predicted home value.")

st.sidebar.header("Property details")

raw_values = {}
raw_values["DEGREE"] = st.sidebar.selectbox("Climate zone (DEGREE)", categorical_options["DEGREE"])
raw_values["METRO3"] = st.sidebar.selectbox("Metro area type (METRO3)", categorical_options["METRO3"])
raw_values["DIVISION"] = st.sidebar.selectbox("Census division", categorical_options["DIVISION"])
raw_values["TYPE"] = st.sidebar.selectbox("Housing type", categorical_options["TYPE"])
raw_values["CONDO"] = st.sidebar.selectbox("Condo?", categorical_options["CONDO"])

built_year = st.sidebar.number_input("Year built", min_value=1800, max_value=date.today().year, value=1990)
raw_values["House_Age"] = date.today().year - built_year

raw_values["PER"] = st.sidebar.number_input("People in household (PER)", min_value=1, max_value=20, value=2)
raw_values["ZADULT"] = st.sidebar.number_input("Adults in household (ZADULT)", min_value=0, max_value=20, value=2)
raw_values["ZINC2"] = st.sidebar.number_input("Household income ($, ZINC2)", min_value=0, value=50000, step=1000)
raw_values["ZSMHC"] = st.sidebar.number_input("Monthly housing cost ($, ZSMHC)", min_value=0, value=1200, step=50)
raw_values["ELDER"] = st.sidebar.number_input("Elderly residents (ELDER)", min_value=0, max_value=10, value=0)
raw_values["BUSPER"] = st.sidebar.number_input("Business/other persons (BUSPER)", min_value=0, max_value=10, value=0)

raw_values["DISH"] = st.sidebar.selectbox("Has dishwasher? (DISH)", categorical_options["DISH"])
raw_values["COOK"] = st.sidebar.selectbox("Has cooking facilities? (COOK)", categorical_options["COOK"])
raw_values["BUYI"] = st.sidebar.selectbox("Recently purchased? (BUYI)", categorical_options["BUYI"])
raw_values["GARAGE"] = st.sidebar.selectbox("Has garage? (GARAGE)", categorical_options["GARAGE"])
raw_values["HLTH"] = st.sidebar.selectbox("Structure health rating (HLTH)", categorical_options["HLTH"])
raw_values["CELLAR"] = st.sidebar.selectbox("Foundation/basement (CELLAR)", categorical_options["CELLAR"])
raw_values["FPLWK"] = st.sidebar.selectbox("Uses fireplace for heat? (FPLWK)", categorical_options["FPLWK"])
raw_values["HHGRAD"] = st.sidebar.selectbox("Householder education (HHGRAD)", categorical_options["HHGRAD"])

st.sidebar.header("Model")
mode = st.sidebar.radio("What do you want to do?", ["Use one model", "Compare all 3 models"])

if mode == "Use one model":
    model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))

run = st.sidebar.button("Predict")

# ------------------------------------------------------------------ main area
if run:
    row = build_input_row(raw_values)

    if mode == "Use one model":
        value = predict(model_choice, row)
        st.subheader(f"Predicted home value — {model_choice}")
        st.metric("Estimated value", f"${value:,.0f}")

        model_metrics = results_df[results_df["Model"] == model_choice].iloc[0]
        st.caption(
            f"Test-set performance for {model_choice}: "
            f"R² = {model_metrics['R2 (log)']:.3f}, "
            f"MAE = ${model_metrics['MAE ($)']:,.0f}"
        )
    else:
        st.subheader("Predicted home value — all models")
        preds = {name: predict(name, row) for name in models}
        pred_df = pd.DataFrame(
            {"Model": list(preds.keys()), "Predicted Value ($)": list(preds.values())}
        ).sort_values("Predicted Value ($)", ascending=False)

        st.bar_chart(pred_df.set_index("Model"))
        st.dataframe(pred_df.style.format({"Predicted Value ($)": "${:,.0f}"}), use_container_width=True)

        st.subheader("Model performance on the test set")
        st.dataframe(
            results_df.style.format(
                {"MAE (log)": "{:.4f}", "RMSE (log)": "{:.4f}", "R2 (log)": "{:.4f}", "MAE ($)": "${:,.0f}"}
            ),
            use_container_width=True,
        )
else:
    st.info("Set the property details in the sidebar, choose a model option, then click **Predict**.")