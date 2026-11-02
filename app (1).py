import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
)

MODEL_DIR = Path("real_estate_models")
CLASSIFIER_PATH = MODEL_DIR / "classification_model.pkl"
REGRESSOR_PATH = MODEL_DIR / "regression_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


@st.cache_resource
def load_assets():
    """Load the two trained models and their metadata once."""
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return classifier, regressor, metadata


def format_inr(value):
    """Format a numeric value as Indian Rupees."""
    return f"₹{value:,.0f}"


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(110deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: #e6f4f7;
        margin: .35rem 0 0 0;
    }
    div[data-testid="stMetric"] {
        background: #f4f7fb;
        border: 1px solid #e1e8f0;
        padding: 14px 16px;
        border-radius: 12px;
    }
    .result-good {
        background: #e8f6ed;
        border-left: 6px solid #168447;
        padding: 15px 18px;
        border-radius: 10px;
        color: #14532d;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .result-caution {
        background: #fff4e5;
        border-left: 6px solid #d97706;
        padding: 15px 18px;
        border-radius: 10px;
        color: #854d0e;
        font-size: 1.2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🏡 Real Estate Investment Advisor</h1>
        <p>Machine-learning-based property investment analysis and five-year price forecast</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Check model files
# ---------------------------------------------------------
required_files = [CLASSIFIER_PATH, REGRESSOR_PATH, METADATA_PATH]
missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error(
        "The application cannot find all required model files. "
        "Place the files in the real_estate_models folder."
    )
    st.code(
        "Real_Estate_Investment_Project/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── real_estate_models/\n"
        "    ├── classification_model.pkl\n"
        "    ├── regression_model.pkl\n"
        "    └── metadata.json"
    )
    st.write("Missing files:")
    for file_name in missing_files:
        st.write(f"- `{file_name}`")
    st.stop()

try:
    classifier, regressor, metadata = load_assets()
except Exception as error:
    st.error(
        "Could not load the saved models. Check that the model files were "
        "created successfully and that the scikit-learn version is compatible."
    )
    st.exception(error)
    st.stop()

features = metadata["features"]
current_year = int(metadata.get("year_reference", 2026))

# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------
st.sidebar.title("🏠 Property Details")
st.sidebar.caption("Enter the property information to run the analysis.")

with st.sidebar.form("property_form"):
    area = st.number_input(
        "Property area (sq. ft.)",
        min_value=100,
        max_value=100000,
        value=2500,
        step=100,
    )
    bedrooms = st.number_input(
        "Bedrooms", min_value=1, max_value=20, value=3, step=1
    )
    bathrooms = st.number_input(
        "Bathrooms", min_value=1, max_value=20, value=2, step=1
    )
    floors = st.number_input(
        "Floors", min_value=1, max_value=100, value=2, step=1
    )
    year_built = st.number_input(
        "Year built",
        min_value=1900,
        max_value=current_year,
        value=min(2018, current_year),
        step=1,
    )
    location_score = st.slider("Location score", 0, 100, 80)
    school_score = st.slider("School score", 0, 100, 75)
    current_price = st.number_input(
        "Current property price (₹)",
        min_value=10000,
        max_value=10000000000,
        value=6500000,
        step=100000,
    )
    submitted = st.form_submit_button(
        "🔍 Analyze Investment",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------
if not submitted:
    st.info(
        "Enter the property details in the left panel and click "
        "**Analyze Investment** to view the prediction dashboard."
    )

    st.subheader("What this application provides")
    intro1, intro2, intro3 = st.columns(3)
    intro1.markdown("**Investment classification**\n\nEstimates whether the property meets the model's good-investment class.")
    intro2.markdown("**Five-year price forecast**\n\nPredicts the property's estimated price after five years.")
    intro3.markdown("**ROI estimate**\n\nCalculates estimated appreciation relative to the current price.")

else:
    # Create exactly the feature columns used during model training.
    property_age = max(0, current_year - int(year_built))
    price_per_sqft = float(current_price) / float(area)

    input_values = {
        "Area_sqft": float(area),
        "Bedrooms": float(bedrooms),
        "Bathrooms": float(bathrooms),
        "Floors": float(floors),
        "Year_Built": float(year_built),
        "Location_Score": float(location_score),
        "School_Score": float(school_score),
        "Current_Price": float(current_price),
        "Property_Age": float(property_age),
        "Price_Per_Sqft": float(price_per_sqft),
    }

    missing_features = [name for name in features if name not in input_values]
    if missing_features:
        st.error(
            "The saved model metadata contains features that this app does "
            f"not currently collect: {', '.join(missing_features)}"
        )
        st.stop()

    input_data = pd.DataFrame(
        [[input_values[name] for name in features]],
        columns=features,
    )

    try:
        prediction = int(classifier.predict(input_data)[0])
        probability = float(classifier.predict_proba(input_data)[0][1])
        future_price = float(regressor.predict(input_data)[0])
    except Exception as error:
        st.error("Prediction failed. Check model compatibility and feature order.")
        st.exception(error)
        st.stop()

    appreciation = future_price - float(current_price)
    roi = (appreciation / float(current_price)) * 100

    st.subheader("📊 Investment Analysis")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric(
        "Current Property Price",
        format_inr(float(current_price)),
    )
    metric2.metric(
        "Estimated Price After 5 Years",
        format_inr(future_price),
        delta=format_inr(appreciation),
    )
    metric3.metric(
        "Estimated Five-Year ROI",
        f"{roi:.2f}%",
    )

    st.divider()
    st.subheader("Investment Recommendation")

    if prediction == 1:
        st.markdown(
            '<div class="result-good">✓ GOOD INVESTMENT — according to the trained classification model</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-caution">! NOT A GOOD INVESTMENT — according to the trained classification model</div>',
            unsafe_allow_html=True,
        )

    st.write(f"**Model probability for the Good Investment class:** {probability * 100:.2f}%")
    st.progress(max(0.0, min(1.0, probability)))

    left, right = st.columns(2)

    with left:
        st.subheader("Five-Year Price Forecast")
        forecast_table = pd.DataFrame(
            {
                "Metric": [
                    "Current property price",
                    "Predicted price after five years",
                    "Expected appreciation",
                    "Estimated five-year ROI",
                ],
                "Model output": [
                    format_inr(float(current_price)),
                    format_inr(future_price),
                    format_inr(appreciation),
                    f"{roi:.2f}%",
                ],
            }
        )
        st.dataframe(forecast_table, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Property Details Submitted")
        details = pd.DataFrame(
            {
                "Input": [
                    "Area (sq. ft.)",
                    "Bedrooms",
                    "Bathrooms",
                    "Floors",
                    "Year built",
                    "Property age",
                    "Location score",
                    "School score",
                    "Price per sq. ft.",
                ],
                "Value": [
                    f"{area:,.0f}",
                    str(bedrooms),
                    str(bathrooms),
                    str(floors),
                    str(year_built),
                    str(property_age),
                    str(location_score),
                    str(school_score),
                    format_inr(price_per_sqft),
                ],
            }
        )
        st.dataframe(details, hide_index=True, use_container_width=True)

    st.download_button(
        "⬇️ Download prediction report (CSV)",
        data=forecast_table.to_csv(index=False).encode("utf-8"),
        file_name="real_estate_investment_prediction.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Disclaimer: This application provides educational machine-learning "
    "estimates, not financial or property investment advice. Actual property "
    "prices may differ. The model does not account for rental yield, taxes, "
    "maintenance, financing costs, or transaction costs."
)
