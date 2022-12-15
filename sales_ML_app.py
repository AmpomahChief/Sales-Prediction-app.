import streamlit as st
import pandas as pd
import os, pickle
import re

# first line after the importation section
st.set_page_config(page_title='Sales Prediction Web app', page_icon="🐞", layout="centered")
DIRPATH = os.path.dirname(os.path.realpath(__file__))


# Loading the Machine learning model
@st.cachec(allow_output_multation = True)
def load_ml_items():
    "Load ML items to reuse them in the app"

    with open('ML_items', 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object

loaded_object = load_ml_items()
if 'results' not in st.session_state:
    st.session_state['results'] = []

# Function to load dataset
@st.chahe()
def load_data(relative_path):
    NewTrain_data = pd.read_csv(relative_path, index_col=0)
    NewTrain_data['onpromotion'] = NewTrain_data['onpromotion'].apply(int)
    NewTrain_data['store_nbr'] = NewTrain_data['store_nbr'].apply(int)
    return NewTrain_data

# Function to extract date items.

def getSeason (row):
    if row in (3, 4, 5):
        return 'Spring'
    elif row in (6, 7, 8): 
        return 'Summer'
    elif row in (9, 10, 11): 
        return 'Autumnf'
    elif row in (12, 1, 2):
        return 'Winter'

def getDateFeatures (NewTrain_data, date):
    NewTrain_data ['date'] = pd.to_datetime(NewTrain_data['date'])
    NewTrain_data ['month'] = NewTrain_data ['date'].dt.month
    NewTrain_data ['day_of_month'] = NewTrain_data ['date'].dt.day 
    NewTrain_data ['day_of_year'] = NewTrain_data['date'].dt.dayofyear
    NewTrain_data ['week_of_year'] = NewTrain_data ['date'].dt.isocalendar().week 
    NewTrain_data ['day_of_week'] = NewTrain_data ['date'].dt.dayofweek
    NewTrain_data ['year'] = NewTrain_data['date'].dt.year
    NewTrain_data ["is_weekend"] = np.where(NewTrain_data ['day_of_week'] > 4, 1, 0)
    NewTrain_data ['is_month_start'] = NewTrain_data ['date'].dt.is_month_start.astype(int) 
    NewTrain_data ['is_month_end'] = NewTrain_data ['date'].dt.is_month_end.astype (int)
    NewTrain_data ['quarter'] = NewTrain_data ['date'].dt.quarter
    NewTrain_data ['is_quarter_start'] = NewTrain_data ['date'].dt.is_quarter_start.astype(int)
    NewTrain_data ['is_quarter_end'] = NewTrain_data['date'].dt.is_quarter_end.astype(int)
    NewTrain_data ['is_year_start'] = NewTrain_data['date'].dt.is_year_start.astype(int)
    NewTrain_data ['is_year_end'] = NewTrain_data['date'].dt.is_year_end.astype (int) 
    NewTrain_data ['season'] = NewTrain_data ['month'].apply(getSeason)

    return NewTrain_data


# Forms to retrive inputs
form = st.form(key = 'information', clear_on_submit = True)

with form:
    # First row
    cols = st.columns((1, 1))
    sales_date = cols[0].date_input('select sales date', min_value = NewTrain_data['sales_date'].min())
    Location = cols[1].selectbox('Please select store location', options = list(NewTrain_data['city'].unique()))
    
    # Second row
    cols = st.columns((1, 1))
    Store_number = cols[0].select_slider('select store number', 1,54 )
    Product_family = cols[1].selectbox('What is the product family', options = list(NewTrain_data['family'].unique()))
    
    # Third row
    cols = st.columns((1, 1))
    Oil_price = cols[0].number_input('Input Current fuel price')
    On_promo = cols[1].select_slider('Is the item on promotion', ['Yes', 'No'])

    # Forth row
    cols = st.columns((1, 1))
    



    # Predict button
    predict = st.form_submit_button(label = 'Make Prediction')