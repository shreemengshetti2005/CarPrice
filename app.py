import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import altair as alt

import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAbmk1egr1Vpu-2qksGJKueoEMuGJkO-4Y")

# Load data and model
df = pd.read_csv('cardekho_dataset.csv')

encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df[['brand','model', 'seller_type', 'fuel_type', 'transmission_type']])
odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
dff = df[['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']]
cdf = pd.concat([odf, dff], axis=1)
cdf.columns = cdf.columns.astype(str)

scaler = StandardScaler()
scaler.fit(cdf)

scaler_y = StandardScaler()
scaler_y.fit(df[['selling_price']])

with open('xgboost_model5.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Home Page
st.title("Know the correct price of your Car!")
st.write("Use this app to predict the selling price of your car based on various parameters.")

# Custom CSS for the slideshow background
st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
        overflow-x: hidden;
    }
    .slideshow-container {
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: -1;
        overflow: hidden;
    }
    .mySlides {
        display: none;
        position: absolute;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .card {
        background: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Custom HTML for the slideshow
st.markdown("""
    <div class="slideshow-container">
        <img class="mySlides" src="https://images.unsplash.com/photo-1488954048779-4d9263af2653?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D">
        <img class="mySlides" src="https://images.unsplash.com/photo-1465929517729-473000af12ce?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D">
    </div>

    <script>
    var slideIndex = 0;
    showSlides();
    function showSlides() {
        var i;
        var slides = document.getElementsByClassName("mySlides");
        for (i = 0; i < slides.length; i++) {
            slides[i].style.display = "none";  
        }
        slideIndex++;
        if (slideIndex > slides.length) {slideIndex = 1}    
        slides[slideIndex-1].style.display = "block";  
        setTimeout(showSlides, 3000); // Change image every 3 seconds
    }
    </script>
""", unsafe_allow_html=True)

# Prediction Section
st.title("Enter the details to predict your car's price")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    brand = st.selectbox("Enter brand", (df['brand'].unique()))
    bbbb = df[df['brand'] == brand]
    model = st.selectbox("Enter Model", bbbb['model'].unique())
    aaaa = df[df['model'] == model]

    aaaa['engine'] = [int(value) for value in aaaa['engine']]
    aaaa['max_power'] = [int(value) for value in aaaa['max_power']]

    vehicle_age = st.number_input("Enter age", value=9)
    km_driven = st.number_input("Enter km driven", value=120000)
    seller_type = 'Individual'
    fuel_type = st.selectbox("Enter fuel type", df['fuel_type'].unique())
    transmission_type = st.selectbox("Enter transmission type", df['transmission_type'].unique())
    mileage = st.number_input("Whats is its mileage (Kmpl)?", value=19.7)
    engine = st.selectbox("Enter Engine capacity (cc)", (aaaa['engine'].unique()))
    max_power = st.selectbox("Max power in BHP",(aaaa['max_power'].unique()))
    seats = st.selectbox("Number of seats",aaaa['seats'].unique())
    st.markdown('</div>', unsafe_allow_html=True)

if vehicle_age < 0:
    st.error("Vehicle age cannot be negative.")
if km_driven < 0:
    st.error("Kilometers driven cannot be negative.")

# Prepare the input data
X = [[brand, model , vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats, 0]]
columns = ['brand','model', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
X = pd.DataFrame(X, columns=columns)

# One-hot encode the categorical features
X_enc = encoder.transform(X[['brand','model', 'seller_type', 'fuel_type', 'transmission_type']])
X_enc = pd.DataFrame.sparse.from_spmatrix(X_enc)

# Combine encoded and numerical features
X_ndf = X[['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']]
X_ccdf = pd.concat([X_enc, X_ndf], axis=1)

# Ensure the column names are string
X_ccdf.columns = X_ccdf.columns.astype(str)

# Standardize the features
X_norm = scaler.transform(X_ccdf)
X_norm = pd.DataFrame(X_norm, columns=X_ccdf.columns)
X_norm = X_norm.drop(columns=['selling_price'])

# Prepare data for prediction
X_for_tree = xgb.DMatrix(X_norm)
y = loaded_model.predict(X_for_tree)

# Inverse transform the predicted value
y = pd.DataFrame(y)
y.columns = ['selling_price']
yd = scaler_y.inverse_transform(y)
yd = pd.DataFrame(yd)
yd.columns = ['selling_price']

# Display prediction result
st.header(f"Predicted Selling Price: ₹{int(yd['selling_price'][0])}")
st.subheader("Note:")
st.write("The predicted price is based on the provided information and market trends. For a more accurate valuation, consider getting an expert inspection.")

# Define initial layout with three columns
col1, col2, col3 = st.columns(3)

# Container for expanded content
expanded_col = st.container()

with col1:
    if st.button("Download Prediction"):
        prediction = int(yd['selling_price'][0])
        prediction_df = pd.DataFrame({
            "Brand": [brand],
            "Model": [model],
            "Predicted Selling Price": [prediction]
        })
        csv = prediction_df.to_csv(index=False)
        expanded_col.download_button(label="Get CSV", data=csv, file_name='prediction.csv', mime='text/csv')

with col2:
    if st.button("Dataset Summary"):
        with expanded_col:
            st.subheader("Dataset Summary")
            st.write(df.describe())

with col3:
    if st.button("Visualizations"):
        with expanded_col:
            st.subheader("Data Visualizations")
            st.write('Visualization of the data we used for model training')

            # Interactive Scatter Plot
            st.subheader("Interactive Scatter Plot")
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x='mileage',
                y='selling_price',
                color='fuel_type',
                tooltip=['brand', 'model', 'mileage', 'selling_price']
            ).interactive()

            st.altair_chart(scatter, use_container_width=True)

            st.subheader("Interactive Scatter Plot")
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x='vehicle_age',
                y='selling_price',
                tooltip=['brand', 'model', 'mileage', 'selling_price']
            ).interactive()

            st.altair_chart(scatter, use_container_width=True)

# Sample Predictions and File Upload

# Sidebar for help
if 'chat_history' not in st.session_state:
    lis = []

input=st.sidebar.text_input("Input: ",key="input")
submit=st.sidebar.button("Ask the question")

if submit and input:
    response=get_gemini_response(input)
    # Add user query and response to session state chat history
    lis.append(("You", input))
    st.sidebar.subheader("The Response is")
    for chunk in response:
        st.sidebar.write(chunk.text)
        lis.append(("Bot", chunk.text))
st.sidebar.subheader("The Chat History is")
    
for role, text in lis:
    st.sidebar.write(f"{role}: {text}")
