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

import google.generativeai as gen_ai


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

with open('xgboost_model_hyper1_lessoverfit.pkl', 'rb') as f:
    loaded_model = pickle.load(f)


st.markdown(
    """
    <style>
    @keyframes slide {
        0% { background: url('https://images.unsplash.com/photo-1488954048779-4d9263af2653?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D') no-repeat center center fixed; }
        25% { background: url('https://images.unsplash.com/photo-1469050061383-f5fd48f3205d?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'); }
        50% { background: url('https://images.unsplash.com/photo-1465929517729-473000af12ce?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D') no-repeat center center fixed; }
        75% { background: url('https://images.unsplash.com/photo-1488954048779-4d9263af2653?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D') no-repeat center center fixed; }
        100% { background: url('https://images.unsplash.com/photo-1517026575980-3e1e2dedeab4?w=400&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fGNhciUyMGRlc2lnbiUyMGJsdWVwcmludHxlbnwwfHwwfHx8MA%3D%3D') no-repeat center center fixed; }
    }

    .main {
      position: relative;
      animation: slide 20s infinite;
      background-size: cover;
      background-position: center;
    }
    
    .main::before {
      content: '';
      position: absolute;
      top: 10;
      left: 10;
      width: 100%;
      height: 370%;
      background: rgba(0, 0, 0, 0.71); /* Adjust opacity to make it darker */
      pointer-events: none;
    }
    
    .content-box {
      background-color: white;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
      margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)



# Main content inside the white box


st.title("Know the correct price of your Car!")
st.write("Use this app to predict the selling price of your car based on various parameters.")
st.title("Enter the details to predict your car's price")

# Prediction form inside the white box
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
y = loaded_model.predict(X_norm)

# Inverse transform the predicted value
y = pd.DataFrame(y)
y.columns = ['selling_price']
yd = scaler_y.inverse_transform(y)
yd = pd.DataFrame(yd)
yd.columns = ['selling_price']

# Display prediction result
st.header(f"Predicted Selling Price: ₹{int(yd['selling_price'][0])}")

st.write("Want to know how to find BHP of your car? Or have doubts regarding vehicle registration? Have a chat with our assistant bot on left.")


#####################################################################################################################


# Sidebar for help
st.sidebar.title("Chat-bot")
# st.sidebar.info("If you have any questions or need assistance, please contact our support team.")

GOOGLE_API_KEY = "AIzaSyCJdcmwF7exDMH0ZvEQl3flYv2DBHgjVqQ"

# Initialize session state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

# Define the button and set the session state
if st.sidebar.button("Send parameters to Gemini AI"):
    st.session_state.button_clicked = True

if st.session_state.button_clicked:
    
    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    modell = gen_ai.GenerativeModel('gemini-pro')

    # Initialize chat session if not already present
    if st.session_state.chat_session is None:
        st.session_state.chat_session = modell.start_chat(history=[])
        # Send the initial context message
        initial_context = f"You are a bot which helps people find correct price for their second hand cars. The car brand is {brand}, the cars model is {model}, the cars age is {vehicle_age}, the cars km driven is {km_driven} and cars predicted selling price is {int(yd['selling_price'][0])}. Just remember this while having further conversation. For now simply greet the user. Also tell user what all you took as input regarding car"
        res = st.session_state.chat_session.send_message(initial_context)

    def translate_role_for_streamlit(user_role):
        if user_role == "model":
            return "Assistant: "
        else:
            return user_role
    # Display the chat history
    for message in st.session_state.chat_session.history:
        if message.parts[0].text != f"You are a bot which helps people find correct price for their second hand cars. The car brand is {brand}, the cars model is {model}, the cars age is {vehicle_age}, the cars km driven is {km_driven} and cars predicted selling price is {int(yd['selling_price'][0])}. Just remember this while having further conversation. For now simply greet the user. Also tell user what all you took as input regarding car":  # Filter out the initial context message
            with st.sidebar:
                st.markdown(translate_role_for_streamlit(message.role))
                st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.sidebar.text_input("Ask Gemini-Pro...")
    if user_prompt:
        # Add user's message to chat and display it
        st.sidebar.write("Me:")
        st.sidebar.markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Gemini-Pro's response
        st.sidebar.write("Bot:")
        st.sidebar.markdown(gemini_response.parts[0].text)

# Button to clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_session = None
    st.session_state.button_clicked = False
    st.sidebar.write("Chat cleared. Start a new conversation.")


st.write()





# Assuming you have your DataFrame 'df' and other variables like 'brand', 'model', 'yd' already defined.

# Define initial layout with three columns for buttons with equal spacing
# cool1 = st.button([])

# # Container for expanded content
# expanded_col = st.container()

# with cool1:
st.subheader("")
st.subheader("Want to buy?")
st.subheader("Find cars under your requirements")
    
cost = st.number_input("Enter your max budget",value=500000)
  
    
        
col1, col2 = st.columns([3, 1])  # Adjust column proportions as needed
with col1:
    vehicle_age_find = st.number_input("Enter maximum age", value=4)
with col2:
    vehicle_age_any_find = st.checkbox("Any", key="age_any")

col3, col4 = st.columns([3, 1])
with col3:
    km_driven_find = st.number_input("Enter max kilometers driven", value=50000)
with col4:
    km_driven_any_find = st.checkbox("Any", key="km_any")
    

# Adjust values based on checkbox status
if vehicle_age_any_find:
    vehicle_age_find = float('inf')  # Exceptionally large value

if km_driven_any_find:
    km_driven_find = float('inf')  # Exceptionally large value

# bbbb = df[df['brand'] == brand]
ddde = df[(df['selling_price'] <= cost) & (df['vehicle_age'] <= vehicle_age_find) & (df['km_driven'] <= km_driven_find)]

ddde = ddde['car_name'].unique()

# ddde = pd.DataFrame(ddde, columns=["Car Name"])

# # Add custom CSS to set the width of the dataframe
# st.markdown(
#     """
#     <style>
#     .dataframe-container {
#         width: 100% !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Use the CSS class on the dataframe
# st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
# st.dataframe(ddde)
# st.markdown('</div>', unsafe_allow_html=True)

emp = "Cars available under these filters are "
lenn = len(ddde)
for i in range(lenn):
    if i != lenn-1:
        emp += f"{ddde[i]}, "
    else:
        emp += f"{ddde[i]}."


st.write(emp)

if st.button("Download List of cars"):
    ddde = pd.DataFrame(ddde, columns=['Car Name'])
    csv = ddde.to_csv(index=False)
    st.download_button(label="Get CSV", data=csv, file_name='car_list.csv', mime='text/csv')



    
    
    
    
    
    

    
    



st.markdown('</div>', unsafe_allow_html=True)




# import streamlit as st
# from google_gemini_pro import gen_ai  # Assuming you have the relevant import for the gemini-pro model



