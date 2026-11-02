
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# Load trained models
classifier = joblib.load(
    "real_estate_models/classification_model.pkl"
)

regressor = joblib.load(
    "real_estate_models/regression_model.pkl"
)

with open("real_estate_models/metadata.json") as f:
    metadata = json.load(f)

features = metadata["features"]
CURRENT_YEAR = metadata["year_reference"]

# Application heading
st.title("🏡 Real Estate Investment Advisor")
st.write(
    "Evaluate property investments using Machine Learning."
)

st.divider()

st.sidebar.header("Enter Property Details")

area = st.sidebar.number_input(
    "Property Area (sq. ft.)", 100, value=2500
)

bedrooms = st.sidebar.number_input(
    "Bedrooms", 1, value=3
)

bathrooms = st.sidebar.number_input(
    "Bathrooms", 1, value=2
)

floors = st.sidebar.number_input(
    "Floors", 1, value=2
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=CURRENT_YEAR,
    value=2018
)

location_score = st.sidebar.slider(
    "Location Score", 0, 100, 80
)

school_score = st.sidebar.slider(
    "School Score", 0, 100, 75
)

current_price = st.sidebar.number_input(
    "Current Property Price",
    min_value=10000,
    value=650000
)

if st.sidebar.button("Analyze Investment", type="primary"):

    property_age = max(0, CURRENT_YEAR - year_built)
    price_per_sqft = current_price / area

    input_data = pd.DataFrame([{
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Year_Built": year_built,
        "Location_Score": location_score,
        "School_Score": school_score,
        "Current_Price": current_price,
        "Property_Age": property_age,
        "Price_Per_Sqft": price_per_sqft
    }])

    input_data = input_data[features]

    # Classification prediction
    prediction = classifier.predict(input_data)[0]
    probability = classifier.predict_proba(input_data)[0][1]

    # Regression prediction
    future_price = regressor.predict(input_data)[0]

    appreciation = future_price - current_price
    roi = (appreciation / current_price) * 100

    st.subheader("Investment Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Property Price",
        f"{current_price:,.2f}"
    )

    col2.metric(
        "Estimated 5-Year Price",
        f"{future_price:,.2f}"
    )

    col3.metric(
        "Estimated ROI",
        f"{roi:.2f}%"
    )

    st.divider()

    st.subheader("Investment Classification")

    if prediction == 1:
        st.success("GOOD INVESTMENT")
    else:
        st.warning("NOT A GOOD INVESTMENT")

    st.write(
        f"Good Investment Probability: {probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.subheader("Five-Year Price Forecast")

    st.write(f"Expected Appreciation: {appreciation:,.2f}")
    st.write(f"Estimated Five-Year ROI: {roi:.2f}%")

    st.caption(
        "Educational machine learning prediction only. "
        "Actual property prices may differ."
    )

else:
    st.info(
        "Enter property details in the sidebar "
        "and click Analyze Investment."
    )
