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

# Apply custom CSS for black background and card styling
st.markdown(
    '''
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stButton button {
        background-color: #00cc00;
        color: black;
    }
    .stButton button:hover {
        background-color: #00ff00;
        color: black;
    }
    .card {
        position: relative;
        width: 100%;
        height: auto;
        background-color: #000;
        display: flex;
        flex-direction: column;
        justify-content: end;
        padding: 12px;
        gap: 12px;
        border-radius: 8px;
        cursor: pointer;
        border: 2px solid #e81cff;
    }

    .card::before {
        content: '';
        position: absolute;
        inset: 0;
        left: -5px;
        margin: auto;
        width: calc(100% + 10px);
        height: calc(100% + 10px);
        border-radius: 10px;
        background: linear-gradient(-45deg, #e81cff 0%, #40c9ff 100% );
        z-index: -10;
        pointer-events: none;
        transition: all 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .card::after {
        content: "";
        z-index: -1;
        position: absolute;
        inset: 0;
        background: linear-gradient(-45deg, #fc00ff 0%, #00dbde 100% );
        transform: translate3d(0, 0, 0) scale(0.95);
        filter: blur(20px);
    }

    .heading {
        font-size: 20px;
        text-transform: capitalize;
        font-weight: 700;
    }

    .card p:not(.heading) {
        font-size: 14px;
    }

    .card p:last-child {
        color: #e81cff;
        font-weight: 600;
    }

    .card:hover::after {
        filter: blur(30px);
    }

    .card:hover::before {
        transform: rotate(-90deg) scaleX(1.34) scaleY(0.77);
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Home Page
st.title("Know the correct price of your Car!")
st.write("Use this app to predict the selling price of your car based on various parameters.")

# Define initial layout with three columns for icons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Dataset Summary"):
        st.subheader("Dataset Summary")
        st.write(df.describe())

with col2:
    if st.button("Visualizations"):
        st.subheader("Data Visualizations")
        st.write('Visualization of the data we used for model training')

        st.subheader("Interactive Scatter Plot")
        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x='mileage',
            y='selling_price',
            color='fuel_type',
            tooltip=['brand', 'model', 'mileage', 'selling_price']
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)

        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x='vehicle_age',
            y='selling_price',
            tooltip=['brand', 'model', 'mileage', 'selling_price']
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)

with col3:
    st.write("Predict your car's price below")

# Prediction Section
st.title("Enter the details to predict your car's price")

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

# Display prediction result in card
st.markdown(
    f'''
    <div class="card">
        <p class="heading">Predicted Selling Price</p>
        <p>₹{int(yd['selling_price'][0])}</p>
        <p>Note:</p>
        <p>The predicted price is based on the provided information and market trends. For a more accurate valuation, consider getting an expert inspection.</p>
    </div>
    ''', 
    unsafe_allow_html=True
)

# Sidebar for help
st.sidebar.subheader("Need Help?")
st.sidebar.info("If you have any questions or need assistance, please contact our support team.")
