#!/usr/bin/env python
# coding: utf-8

# # Importing libraries:

# In[3]:


import streamlit as st
import pickle
import pandas as pd
import os


# ## Importing models:

# In[5]:

lr_model = pickle.load(open('model_dep/model_lr.pkl', 'rb'))
scaler = pickle.load(open('model_dep/scaler.pkl', 'rb'))
encoder = pickle.load(open('model_dep/encorder.pkl', 'rb'))


# ## Deployment:

# In[ ]:


# App title
st.title("Cluster Prediction")

# User input form
st.header("Enter Customer Details")
age = st.number_input("Age", min_value=10, max_value=100, value=30)
education = st.selectbox("Education Level", ["Basic", "Graduation", "Master", "PhD", "2n Cycle"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Together", "Divorced", "Widow", "Alone", "Absurd", "YOLO"])
income = st.number_input("Income", min_value=0, step=1000)
kidhome = st.number_input("No. of children in customer's household", min_value=0, max_value=10, step=1)
teenhome = st.number_input("No. of teenagers in customer's household", min_value=0, max_value=10, step=1)
recency = st.number_input("No. of days since customer's last purchase", min_value=0, max_value=100, step=1)
wines = st.number_input("Amount spent on wine in last 2 years", min_value=0, max_value=1500)
fruits = st.number_input("Amount spent on fruits", min_value=0, max_value=200)
meatproducts = st.number_input("Amount spent on meat", min_value=0, max_value=1800)
fishproducts = st.number_input("Amount spent on fish", min_value=0, max_value=300)
sweetproducts = st.number_input("Amount spent on sweet", min_value=0, max_value=300)
goldproducts = st.number_input("Amount spent on gold", min_value=0, max_value=400)
dealspurchases = st.number_input("No. of purchases made with a discount", min_value=0, max_value=15, step=1)
webpurchases = st.number_input("purchases made through the website", min_value=0, max_value=30, step=1)
catalogpurchases = st.number_input("purchases made using a catalogue", min_value=0, max_value=30, step=1)
storepurchases = st.number_input("purchases made directly in stores", min_value=0, max_value=15, step=1)
webvisitsmonth = st.number_input("visits to website in the last month", min_value=0, max_value=20, step=1)
# Campaign acceptance:
st.subheader("Select one campaign:")
acceptedcmp3 = st.radio("1 if customer accepted the offer in the 3rd campaign, 0 otherwise", [0, 1], index=0)
acceptedcmp4 = st.radio("1 if customer accepted the offer in the 4th campaign, 0 otherwise", [0, 1], index=0)
acceptedcmp5 = st.radio("1 if customer accepted the offer in the 5th campaign, 0 otherwise", [0, 1], index=0)
acceptedcmp1 = st.radio("1 if customer accepted the offer in the 1st campaign, 0 otherwise", [0, 1], index=0)
acceptedcmp2 = st.radio("1 if customer accepted the offer in the 2nd campaign, 0 otherwise", [0, 1], index=0)
st.subheader("Complain:")
complain = st.radio("1 if the customer complained in the last 2 years, 0 otherwise", [0, 1], index=0)
st.subheader("Response:")
response = st.radio("1 if customer accepted the offer in the last campaign, 0 otherwise", [0, 1], index=0)

# Determine age group from age
def get_age_group(age):
    if age <= 12:
        return 'Child'
    elif 13 <= age <= 24:
        return 'Young'
    elif 25 <= age <= 59:
        return 'Adult'
    else:
        return 'Old'

age_group = get_age_group(age)
st.markdown(f"**Automatically assigned Age Group:** `{age_group}`")

# Preprocessing (based on training)
# converting categorical to numerical:
education, marital_status, age_group = encoder.transform([[education, marital_status, age_group]])[0]
# Scalling data:
income, recency, wines, fruits, meatproducts, fishproducts, sweetproducts, goldproducts = scaler.transform([[income, recency, wines, fruits, meatproducts, fishproducts, sweetproducts, goldproducts]])[0]

# Combine all inputs
customer_data = {
    "Age": age,
    "Education": education,
    "Marital_Status": marital_status,
    "Income": income,
    "Kidhome": kidhome,
    "Teenhome": teenhome,
    "Recency": recency,
    "MntWines": wines,
    "MntFruits": fruits,
    "MntMeatProducts": meatproducts,
    "MntFishProducts": fishproducts,
    "MntSweetProducts": sweetproducts,
    "MntGoldProducts": goldproducts,
    "NumDealsPurchases": dealspurchases,
    "NumWebPurchases": webpurchases,
    "NumCatalogPurchases": catalogpurchases,
    "NumStorePurchases": storepurchases,
    "NumWebVisitsMonth": webvisitsmonth,
    "AcceptedCmp3": acceptedcmp3,
    "AcceptedCmp4": acceptedcmp4,
    "AcceptedCmp5": acceptedcmp5,
    "AcceptedCmp1": acceptedcmp1,
    "AcceptedCmp2": acceptedcmp2,
    "Complain": complain,
    "Response": response,
    "Age_group": age_group
}

st.subheader("Encorded and scalled Customer Data:")
st.json(customer_data)

customer_data = list(customer_data.values())

# Make prediction
if st.button("Predict"):
    prediction = lr_model.predict([customer_data])
    st.success(f"This customer data is comes under : {prediction[0]} cluster.")


# In[ ]:




