import streamlit as st
import pandas as pd
import joblib
# ==========================
# Load Saved Files
# ==========================
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
similarity_df = joblib.load("recommendations.pkl")
# ==========================
# Streamlit UI
# ==========================
st.set_page_config(
    page_title="Customer Segmentation & Recommendation",
    layout="wide"
)
st.title("🛒 Customer Segmentation & Product Recommendation System")
option = st.sidebar.radio(
    "Select Module",
    ["Customer Segmentation", "Product Recommendation"]
)
# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================
if option == "Customer Segmentation":
    st.header("Customer Segment Prediction")
    recency = st.number_input(
        "Recency (Days Since Last Purchase)",
        min_value=0,
        value=10
    )
    frequency = st.number_input(
        "Frequency (Number of Purchases)",
        min_value=0,
        value=5
    )
    monetary = st.number_input(
        "Monetary Value (Total Spending)",
        min_value=0.0,
        value=1000.0
    )
    if st.button("Predict Cluster"):
        input_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        st.success(
            f"Customer belongs to Cluster {cluster}"
        )
# =====================================================
# PRODUCT RECOMMENDATION
# =====================================================
elif option == "Product Recommendation":
    st.header("Product Recommendation System")
    product = st.selectbox(
        "Select Product",
        similarity_df.columns
    )
    if st.button("Recommend Products"):
        recommendations = (
            similarity_df[product]
            .sort_values(ascending=False)
            .iloc[1:6]
        )
        st.subheader("Recommended Products")
        for item in recommendations.index:
            st.write("✅", item)