import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration and Data Loading ---
st.set_page_config(page_title="Product Return Risk Predictor", layout="wide")

# Load the trained CatBoost model
try:
    model = joblib.load('catboost_model.joblib')
except FileNotFoundError:
    st.error("Model file 'catboost_model.joblib' not found. Please ensure it's in the same directory.")
    st.stop()

# Define feature lists based on training data (assuming X_train from previous steps)
numerical_features = ['Age', 'Quantity', 'Price', 'Discount', 'Product Rating', 'Total_Amount', 'Effective_Price']

category_cols = ['Dresses', 'Ethnic Wear', 'Formal Wear', 'Jackets', 'Jeans', 'Shirts', 'Shorts', 'Skirts', 'Sleepwear', 'Suits', 'Sweaters', 'T-shirts', 'Trousers', 'Undergarments']
brand_cols = ['H&M', "Levie's", 'Nike', 'Pantaloons', 'Puma', 'Raymond', 'Zudio']

# Define states for each geographical region
east_states = ['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Tripura', 'West Bengal']
west_states = ['Goa', 'Gujarat', 'Maharashtra']
central_states = ['Chhattisgarh', 'Madhya Pradesh']
south_states = ['Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana']  # Note: Andhra Pradesh is removed as it was not in the original list of existing states from df.columns
north_states = ['Bihar', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Uttar Pradesh', 'Uttarakhand']

all_states = sorted(list(set(east_states + west_states + central_states + south_states + north_states)))

# Expected columns for the model input (order matters for some models, but CatBoost handles it better)
# This list should match X.columns at the time of model training
expected_columns = ['Age', 'Gender', 'Quantity', 'Price', 'Discount', 'Product Rating',
                    'Category_Dresses', 'Category_Ethnic Wear', 'Category_Formal Wear', 'Category_Jackets', 'Category_Jeans',
                    'Category_Shirts', 'Category_Shorts', 'Category_Skirts', 'Category_Sleepwear', 'Category_Suits',
                    'Category_Sweaters', 'Category_T-shirts', 'Category_Trousers', 'Category_Undergarments',
                    'Brand_H&M', "Brand_Levie's", 'Brand_Nike', 'Brand_Pantaloons', 'Brand_Puma', 'Brand_Raymond', 'Brand_Zudio',
                    'Total_Amount', 'Effective_Price', 'payment_mode_Online Payment',
                    'Region_East', 'Region_West', 'Region_Central', 'Region_South', 'Region_North']

# --- Streamlit UI ---
st.title("🛍️ Product Return Risk Predictor")
st.markdown("Enter the product and customer details to predict the risk of return.")

with st.form("prediction_form"):
    st.header("Customer & Product Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 99, 30)
        gender = st.radio("Gender", options=['Female', 'Male'], index=0)
        quantity = st.number_input("Quantity", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        price = st.number_input("Price", min_value=0.0, value=500.0, step=10.0)

    with col2:
        discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        product_rating = st.slider("Product Rating", 0.0, 5.0, 3.5, step=0.5)

        # --- Auto Calculations for Total Amount & Effective Price ---
        total_amount = quantity * price
        # Assuming discount is given in percentage
        effective_price = total_amount * (1 - discount / 100.0)

        st.markdown("### Auto-calculated values")
        st.write(f"**Total Amount:** {total_amount:.2f}")
        st.write(f"**Effective Price:** {effective_price:.2f}")

    with col3:
        selected_category = st.selectbox("Product Category", options=category_cols, index=0)
        selected_brand = st.selectbox("Brand", options=brand_cols, index=0)
        payment_mode = st.radio("Payment Mode", options=['Cash on Delivery', 'Online Payment'], index=0)
        selected_state = st.selectbox("State", options=all_states, index=0)

    submitted = st.form_submit_button("Predict Return Risk")

    if submitted:
        # --- Feature Engineering for Input ---
        input_data = {}

        # Numerical features
        input_data['Age'] = age
        input_data['Quantity'] = quantity
        input_data['Price'] = price
        input_data['Discount'] = discount
        input_data['Product Rating'] = product_rating
        input_data['Total_Amount'] = total_amount
        input_data['Effective_Price'] = effective_price

        # Gender (0 for Female, 1 for Male based on original data)
        input_data['Gender'] = 1 if gender == 'Male' else 0

        # One-hot encode Category
        for cat in category_cols:
            input_data[f'Category_{cat}'] = (selected_category == cat)

        # One-hot encode Brand
        for brand in brand_cols:
            # Handle Brand_Levie's special character
            if brand == "Levie's":
                input_data[f"Brand_Levie's"] = (selected_brand == brand)
            else:
                input_data[f'Brand_{brand}'] = (selected_brand == brand)

        # Payment Mode
        input_data['payment_mode_Online Payment'] = (payment_mode == 'Online Payment')

        # Region mapping
        input_data['Region_East'] = (selected_state in east_states)
        input_data['Region_West'] = (selected_state in west_states)
        input_data['Region_Central'] = (selected_state in central_states)
        input_data['Region_South'] = (selected_state in south_states)
        input_data['Region_North'] = (selected_state in north_states)
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Ensure all expected columns are present, fill missing with False/0 and reorder
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = False  # For boolean features, default to False
        
        input_df = input_df[expected_columns]  # Reorder columns to match training data
        
        # Convert boolean columns to int if the model expects numerical input (CatBoost handles bools, but good practice)
        for col in input_df.columns:
            if input_df[col].dtype == 'bool':
                input_df[col] = input_df[col].astype(int)

        # --- Make Prediction ---
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of high return risk (class 1)

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.error(f"**High Return Risk!** (Probability: {prediction_proba:.2%})")
            st.write("This product is likely to be returned based on the provided details.")
        else:
            st.success(f"**Low Return Risk.** (Probability: {prediction_proba:.2%})")
            st.write("This product has a low risk of being returned.")

        st.markdown("---")
        st.subheader("Input Features for Prediction:")
        st.dataframe(input_df)
