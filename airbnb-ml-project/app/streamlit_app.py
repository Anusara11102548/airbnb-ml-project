import streamlit as st
import joblib
import numpy as np

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# โหลดโมเดล
model = joblib.load("../model/price_model.pkl")

# ส่วนหัว
st.markdown(
    """
    <h1 style='text-align: center; color: #FF5A5F;'>🏡 Airbnb Price Predictor</h1>
    <p style='text-align: center;'>Predict the estimated price of an Airbnb listing using Machine Learning</p>
    """,
    unsafe_allow_html=True
)

st.write("")

# แบ่ง layout
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📋 Listing Information")

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=500,
        value=50
    )

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        max_value=20.0,
        value=1.5
    )

    availability_365 = st.number_input(
        "Availability (days per year)",
        min_value=0,
        max_value=365,
        value=200
    )

    predict_button = st.button("💰 Predict Price")

with col2:
    st.subheader("📊 Prediction Result")

    if predict_button:

        X = np.array([[number_of_reviews, reviews_per_month, availability_365]])

        prediction = model.predict(X)

        st.success(f"Estimated Airbnb Price: ${prediction[0]:.2f}")

        st.balloons()

    else:
        st.info("Enter listing details and click Predict Price")

st.divider()

# ส่วนล่าง
st.markdown(
    """
    <center>
    <p style='color: gray'>
    Machine Learning Deployment Project<br>
    Built with Streamlit & Scikit-learn
    </p>
    </center>
    """,
    unsafe_allow_html=True
)
