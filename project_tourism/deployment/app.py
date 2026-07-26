import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# --- Streamlit App Configuration (Must be first Streamlit command) --- #
st.set_page_config(page_title="Tourism Package Predictor", layout="centered")

# --- Paths --- #
MODEL_PATH = "project_tourism/deployment/best_tourism_prediction_model.joblib"
DATA_PATH = "project_tourism/data/tourism.csv"

# --- Load Model and Data for Encoders --- #
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load(MODEL_PATH)
        df_original = pd.read_csv(DATA_PATH)

        # Fit LabelEncoders on original data to ensure consistency
        encoders = {}
        categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation']
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(df_original[col])
            encoders[col] = le
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"Error loading essential files: {e}. Make sure 'model' and 'data' directories exist and contain the model and original data.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during loading: {e}")
        st.stop()

model, encoders = load_model_and_encoders()

# --- Streamlit App Content --- #
st.title("🌴 Tourism Package Purchase Predictor ✈️")
st.markdown("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# --- Input Form --- #
with st.form("prediction_form"):
    st.header("Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Age", 18, 70, 30)
        TypeofContact = st.selectbox("Type of Contact", list(encoders['TypeofContact'].classes_))
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        Occupation = st.selectbox("Occupation", list(encoders['Occupation'].classes_))
        Gender = st.selectbox("Gender", list(encoders['Gender'].classes_))
        NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 1)

    with col2:
        PreferredPropertyStar = st.slider("Preferred Property Star", 1, 5, 3)
        MaritalStatus = st.selectbox("Marital Status", list(encoders['MaritalStatus'].classes_))
        NumberOfTrips = st.number_input("NumberOfTrips Annually", 0, 20, 2)
        Passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        OwnCar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5)", 0, 5, 0)

    st.header("Interaction Details")
    col3, col4 = st.columns(2)

    with col3:
        Designation = st.selectbox("Designation", list(encoders['Designation'].classes_))
        MonthlyIncome = st.number_input("Monthly Income", 0, 100000, 30000)
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    with col4:
        ProductPitched = st.selectbox("Product Pitched", list(encoders['ProductPitched'].classes_))
        NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
        DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 15)

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create DataFrame from inputs
        input_data = pd.DataFrame({
            'Age': [Age],
            'TypeofContact': [TypeofContact],
            'CityTier': [CityTier],
            'Occupation': [Occupation],
            'Gender': [Gender],
            'NumberOfPersonVisiting': [NumberOfPersonVisiting],
            'PreferredPropertyStar': [PreferredPropertyStar],
            'MaritalStatus': [MaritalStatus],
            'NumberOfTrips': [NumberOfTrips],
            'Passport': [Passport],
            'OwnCar': [OwnCar],
            'NumberOfChildrenVisiting': [NumberOfChildrenVisiting],
            'Designation': [Designation],
            'MonthlyIncome': [MonthlyIncome],
            'PitchSatisfactionScore': [PitchSatisfactionScore],
            'ProductPitched': [ProductPitched],
            'NumberOfFollowups': [NumberOfFollowups],
            'DurationOfPitch': [DurationOfPitch]
        })

        # Apply Label Encoding
        for col, encoder in encoders.items():
            # Handle cases where input might not be in training categories
            # This assumes `handle_unknown='ignore'` was effectively used or not an issue during training
            # For Streamlit, we convert unseen labels to the most frequent or a placeholder
            # For simplicity, if a label is not found, it will raise an error. Ensure original data covers all options.
            input_data[col] = encoder.transform(input_data[col])

        # Make prediction
        prediction_proba = model.predict_proba(input_data)[:, 1]
        prediction_class = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        if prediction_class == 1:
            st.success(f"The customer is predicted to **purchase** the package with a probability of {prediction_proba[0]:.2f}.")
        else:
            st.info(f"The customer is predicted **not to purchase** the package with a probability of {1 - prediction_proba[0]:.2f}.")

        st.write("--- Debug Info ---")
        st.write("Processed Input Data:")
        st.write(input_data)
