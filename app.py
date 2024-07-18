import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

st.title("Know the correct price of your Car!")
df = pd.read_csv('cardekho_dataset.csv')

encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df[['brand','model','seller_type','fuel_type','transmission_type']])

dff = df[['vehicle_age','km_driven','mileage','engine','max_power','selling_price']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(dff)

scaler_y = MinMaxScaler(feature_range=(0, 1))
# Fit and transform the target variable
scaler_y.fit(df[['selling_price']])



# Load the model from file
with open('xgboost_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use the loaded model for predictions or further analysis
# brands = pd.Series(df['brand'].unique())
brand = st.selectbox("Enter brand",(df['brand'].unique()))

model = st.selectbox("Enter Model",(df['model'].unique()))

vehicle_age = st.number_input("Enter age",value=9)

km_driven = st.number_input("Enter km driven",value=120000)

seller_type = st.selectbox("Enter seller type",df['seller_type'].unique())

fuel_type = st.selectbox("Enter fuel type",df['fuel_type'].unique())

transmission_type = st.selectbox("Enter transmission type",df['transmission_type'].unique())

mileage = st.number_input("Whats is its mileage (Kmpl)?",value=19.7)

engine = st.number_input("Engine capacity in CC (Cubic Centimeters)",value=796)

max_power = st.number_input("Max power in BHP",value=46.3)

seats = st.number_input("number of seats",1,9,value=5)
selling_price = 0
############################################################


X = [[brand,model,vehicle_age,km_driven,seller_type,fuel_type,transmission_type,mileage,engine,max_power,seats,selling_price]]

columns = ['brand', 'model', 'vehicle_age', 'km_driven', 'seller_type',
           'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power', 'seats','selling_price']

X = pd.DataFrame(X,columns=columns)

X_enc = encoder.transform(X[['brand','model','seller_type','fuel_type','transmission_type']])

Xedf = pd.DataFrame.sparse.from_spmatrix(X_enc)

Xndf = X[['vehicle_age','km_driven','mileage','engine','max_power','selling_price']]

X_norm = scaler.transform(Xndf)

X_norm = pd.DataFrame(X_norm, columns=Xndf.columns)
X_norm = X_norm.drop(columns=['selling_price'])

Xcdf = pd.concat([Xedf,X_norm],axis = 1)

X_for_tree = xgb.DMatrix(Xcdf)
y = loaded_model.predict(X_for_tree)

y = pd.DataFrame(y)
y.columns = ['selling_price']
yd = scaler_y.inverse_transform(y)
yd = pd.DataFrame(yd)
yd.columns = ['selling_price']

st.write(yd)
