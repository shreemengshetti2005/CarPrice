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

# flagg = False
    
# if flagg == False:
#     lis = []
#     flagg = True

import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyAbmk1egr1Vpu-2qksGJKueoEMuGJkO-4Y")

# model = genai.GenerativeModel('gemini-1.5-flash')



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

# Display prediction result
st.header(f"Predicted Selling Price: ₹{int(yd['selling_price'][0])}")
st.subheader("Note:")
st.write("The predicted price is based on the provided information and market trends. For a more accurate valuation, consider getting an expert inspection.")

# Define initial layout with three columns
col1, col2, col3 = st.columns(3)

# Container for expanded content
expanded_col = st.container()

with col1:
    # prediction = int(yd['selling_price'][0])
    # prediction_df = pd.DataFrame({
    #     "Brand": [brand],
    #     "Model": [model],
    #     "Predicted Selling Price": [prediction]
    # })
    # csv = prediction_df.to_csv(index=False)
    # expanded_col.download_button(label="Download Prediction", data=csv, file_name='prediction.csv', mime='text/csv')
    
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

            # # Histogram of vehicle ages
            # fig, ax = plt.subplots()
            
            # ax.hist(df['vehicle_age'], bins=20, color='blue', alpha=0.7)
            # ax.set_xlabel('Vehicle Age (years)')
            # ax.set_ylabel('Frequency')
            
            # st.pyplot(fig)

            # st.subheader("Interactive Scatter Plot")
            # scatter = alt.Chart(df).mark_circle(size=60).encode(
            #     x='transmission_type',
            #     y='selling_price',
            #     # color='fuel_type',
            #     tooltip=['brand', 'model', 'mileage', 'selling_price']
            # ).interactive()

            # st.altair_chart(scatter, use_container_width=True)


            # # Scatter plot of mileage vs selling price
            # fig, ax = plt.subplots()
            # ax.scatter(df['mileage'], df['selling_price'], alpha=0.5)
            # ax.set_xlabel('Mileage (Kmpl)')
            # ax.set_ylabel('Selling Price')
            # st.pyplot(fig)

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
                # color='fuel_type',
                tooltip=['brand', 'model', 'mileage', 'selling_price']
            ).interactive()

            st.altair_chart(scatter, use_container_width=True)

# Sample Predictions and File Upload

# Sidebar for help
# st.sidebar.subheader("Need Help?")
# st.sidebar.info("If you have any questions or need assistance, please contact our support team.")

# import streamlit as st
# import os
# import google.generativeai as genai

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to load Gemini Pro model and get repsonses
model=genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])

def get_gemini_response(question):
    
    response=chat.send_message(question,stream=True)
    return response

##initialize our streamlit app

# stset_page_config(page_title="Q&A Demo")

# st.sidebar("Gemini LLM Application")

# Initialize session state for chat history if it doesn't exist
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
    




    

