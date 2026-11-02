import joblib
import json
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Real Estate Investment Advisor", page_icon="🏡", layout="wide"
)

# Load trained models
classifier = joblib.load("real_estate_models/classification_model.pkl")
regressor = joblib.load("real_estate_models/regression_model.pkl")

with open("real_estate_models/metadata.json") as f:
    metadata = json.load(f)

features = metadata["features"]
CURRENT_YEAR = metadata["year_reference"]

# Application heading
st.title("🏡 Real Estate Investment Advisor")
st.write("Evaluate property investments using Machine Learning.")
st.divider()

st.sidebar.header("Enter Property Details")
area = st.sidebar.number_input("Property Area (sq. ft.)", min_value=100, value=2500)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, value=2)
floors = st.sidebar.number_input("Floors", min_value=1, value=2)
year_built = st.sidebar.number_input(
    "Year Built", min_value=1900, max_value=CURRENT_YEAR, value=2018
)
location_score = st.sidebar.slider("Location Score", 0, 100, 80)
school_score = st.sidebar.slider("School Score", 0, 100, 75)
current_price = st.sidebar.number_input(
    "Current Property Price", min_value=10000, value=650000
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
      "Price_Per_Sqft": price_per_sqft,
  }])

  # Ensure columns match the exact training order
  input_data = input_data[features]

  # Classification prediction
  prediction = classifier.predict(input_data)[0]
  # Assuming binary classification (e.g., Good vs. Bad investment, or Buy vs. Don't Buy)
  try:
    probability = classifier.predict_proba(input_data)[0][prediction]
  except AttributeError:
    probability = None

  # Regression prediction (e.g., predicted future price or fair value)
  predicted_value = regressor.predict(input_data)[0]

  # Display results
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Investment Recommendation")
    if prediction == 1:  # Adjust label logic based on your encoder/target mapping
      st.success(
          f"**Recommended Investment** (Confidence: {probability*100:.1f}%)"
          if probability
          else "**Recommended Investment**"
      )
    else:
      st.error(
          f"**Not Recommended** (Confidence: {probability*100:.1f}%)"
          if probability
          else "**Not Recommended**"
      )

  with col2:
    st.subheader("Model Valuation")
    st.metric(
        label="Estimated Property Value",
        value=f"${predicted_value:,.2f}",
        delta=f"${predicted_value - current_price:,.2f} vs Current",
    )
