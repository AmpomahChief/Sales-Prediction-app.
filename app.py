 # importing librabries
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import re
from sklearn import preprocessing
from PIL import Image


# first line after the importation section
st.set_page_config(page_title="Sales Prediction app", page_icon='💲', layout="centered", initial_sidebar_state = 'auto')
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Setting the page title
st.title("Sales Prediction Web Application")


# Loading Machine Learning items
@st.cache(allow_output_mutation=True)
def Load_ml_items(relative_path):
    "Load ML items to reuse them"
    with open(relative_path, 'rb' ) as file:
        loaded_object = pickle.load(file)
    return loaded_object


Loaded_object = Load_ml_items('assets/ML_items.pkl')

# Loaded_object = Load_ml_items('ML_items.pl')
model, encoder, data = Loaded_object['model'], Loaded_object['encoder'], Loaded_object['data']


# Function to get date features from the inputs
@st.cache(allow_output_mutation=True)
def getSeason (row):
    if row in (3, 4, 5):
        return 'Spring'
    elif row in (6, 7, 8):
        return 'Summer'
    elif row in (9, 10, 11):
        return 'Autumnf'
    elif row in (12, 1, 2):
        return 'Winter'

def getDateFeatures (df,date):
    df ['date'] = pd.to_datetime(df['date'])
    df ['month'] = df ['date'].dt.month
    df ['day_of_month'] = df ['date'].dt.day 
    df ['day_of_year'] = df['date'].dt.dayofyear
    df ['week_of_year'] = df ['date'].dt.isocalendar().week 
    df ['day_of_week'] = df ['date'].dt.dayofweek
    df ['year'] = df['date'].dt.year
    df ["is_weekend"] = np.where(df ['day_of_week'] > 4, 1, 0)
    df ['is_month_start'] = df ['date'].dt.is_month_start.astype(int) 
    df ['is_month_end'] = df ['date'].dt.is_month_end.astype (int)
    df ['quarter'] = df ['date'].dt.quarter
    df ['is_quarter_start'] = df ['date'].dt.is_quarter_start.astype(int)
    df ['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df ['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df ['is_year_end'] = df['date'].dt.is_year_end.astype (int) 
    df ['season'] = df ['month'].apply(getSeason)

    return df
   
   # Icon for the page
image = Image.open("assets/groceries.jpeg")
st.image(image, width = 600)

# Creating elements of the sidebat
st.sidebar.header("This Web App is a deployment of a machine model that predicts unit sales")
check =st.sidebar.checkbox("Column discription")
if check:
    st.sidebar.markdown(""" 
                    - **STORE_NBR** identifies the store at which the products are sold.
                    - **FAMILY** identifies the type of product sold.
                    - **SALES** this is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **ONPROMOTION** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **DATA** is the date on which a transaction / sale was made
                    - **LOCATION** is the city in which the store is located
                    - **TYPE** is the type of store, based on Corporation Favorita's own type system
                    - **oIL_PRICE** is the daily oil price
                    - **HOLIDAY_TYPE** indicates whether the day was a holiday, event day, or a workday
                    - **LOCALE** indicates whether the holiday was local, national or regional.
                    - **TRANSFERRED** indicates whether the day was a transferred holiday or not.
                    """)

 # # Setting up variables for input data
@st.cache()
def setup(tmp_df_file):
    "Setup the required elements like files, models, global variables, etc"
    pd.DataFrame (
        dict(
            date=[],
            store_nbr=[],
            family=[],
            onpromotion=[],
            city=[],
            locale=[],
            type_of_day=[],
            oil_price=[],
        )
    ).to_csv(tmp_df_file, index=False)

# Setting up a file to save our input data
tmp_df_file = os.path.join(DIRPATH, "tmp", "data.csv")
setup(tmp_df_file)

##################################################################

# Forms to retrieve input
form = st.form(key="information", clear_on_submit=True)

with form:
    cols = st.columns((1, 1))
    cols = st.columns(2)
    date = cols[0].date_input('select sales date', min_value = data['date'].min())
    Location = cols[1].selectbox('Please select store location', options = list(data['city'].unique()))
    
    # Second row
    Store_number = cols[0].select_slider('select store number', options = list(data['store_nbr'].unique()))
    Product_family = cols[1].selectbox('What is the product family', options = list(data['family'].unique()))
    
    # Third row
    oil_price = cols[0].number_input('Input Current fuel price')
    # On_promo = cols3[1].select_slider('Is the item on promotion', ['Yes', 'No'])
    on_promo = cols[1].slider("Select number of items on promo:",min_value =0, max_value = 742,step =1)
    

    # Forth row   
    st.markdown('**ADDITIONAL INFO**')
    check = cols[0].checkbox('Is it a Holiday or a weekend')
    if check:
            Type = cols[0].selectbox('Holiday type', options=(data["type_of_day"].unique()))
            locale = cols[1].selectbox('Locale:', options=(data["locale"].unique()))
            
    else:
        Type= 'Work Day'
        locale = 'National'      
       
       
    # Submit button
    submitted = st.form_submit_button(label= "Make Prediction")

    
    ##############################################################################
if submitted:
    st.success('Form Recieved!', icon="✔️")  
        # Inputs formatting
    pd.read_csv(tmp_df_file).append(
        dict(
                date=date, 
                store_nbr=Store_number,   
                family= Product_family,
                onpromotion=on_promo,
                city=Location,
                locale=locale,
                type_of_day=Type,
                oil_price=oil_price,
            ),
                ignore_index=True,
    ).to_csv(tmp_df_file, index=False)
    st.balloons()
    
    df = pd.read_csv(tmp_df_file)
    input_df = df.copy()
#####################################################################

    # Extracting date Features
    processed_data = getDateFeatures(df, 'date')
    
    # Deleting the date column.
    processed_data = processed_data.drop(columns=['date'])

    # Encoding Categorical Variables
    encoder = preprocessing.LabelEncoder()

    processed_data['season'] = encoder.fit_transform(processed_data['season'])
    processed_data['family'] = encoder.fit_transform(processed_data['family'])
    processed_data['city'] = encoder.fit_transform(processed_data['city'])
    processed_data['type_of_day'] = encoder.fit_transform(processed_data['type_of_day'])
    processed_data['locale'] = encoder.fit_transform(processed_data['locale'])
    
     def predict(X, model):
        results = model.predict(X)
        return results
    
    prediction = predict(processed_data, model)
    df['sales']= prediction 
    
    
    # Displaying prediction results
    st.markdown('''---''')
    st.markdown("<h4 style='text-align: center;'> Prediction Results </h4> ", unsafe_allow_html=True)
    st.success(f"Predicted Sales: {prediction[-1]}")
    st.markdown('''---''')

    # Making expander to view all records
    expander = st.expander("See all records")
    with expander:
        df = pd.read_csv(tmp_df_file)
        df['sales']= prediction
        st.dataframe(df)
   
