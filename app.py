 # importing librabries
import streamlit as st
import pandas as pd
import numpy as np
import os
import os, pickle
import re
from PIL import Image


# first line after the importation section
st.set_page_config(page_title="Demo app", page_icon="🐞", layout="centered", layout = 'wide', initial_sidebar_state = 'auto')
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Setting the page title
st.title("Sales Prediction Web Application")
# ---- Importing and creating other key elements items
# Importing machine learning toolkit

# DIRPATH = os.path.dirname(__file__)
# ASSETSDIRPATH = os.path.join(DIRPATH, 'assets')
# ML_ITEMS = os.path.join(ASSETSDIRPATH, 'ML_items.pkl')
# DATASET = os.path.join(ASSETSDIRPATH, 'NewTrian.csv')

@st.cache(allow_output_mutation=True)
def load_ml_toolkit(relative_path):
    with open(relative_path, "rb") as file:
        loaded_object = pickle.load(file)
    return loaded_object

# Function to load the dataset
@st.cache()
def load_data(relative_path):
    NewTrain = pd.read_csv(relative_path, index_col= 0)
    NewTrain["onpromotion"] = NewTrain["onpromotion"].apply(int)
    NewTrain["store_nbr"] = NewTrain["store_nbr"].apply(int)
    NewTrain["date"] = pd.to_datetime(NewTrain["date"]).dt.date
    # NewTrain["year"]= pd.to_datetime(NewTrain['date']).dt.year
    return NewTrain

# ----- Loading key components
# Loading the base dataframe

# rpath = '/Users/Bernard Ampomah/Desktop/P4 ML app/assets/NewTrain.csv'
NewTrain = load_data("/Users/Bernard Ampomah/Desktop/BAP Accelerator Projects/Projects/LP 2 Time Series/NewTrain.csv")

# Loading the toolkit
loaded_toolkit = load_ml_toolkit("/Users/Bernard Ampomah/Desktop/BAP Accelerator Projects/Projects/LP 2 Time Series/ML_items.pkl")
if "results" not in st.session_state:
    st.session_state["results"] = []

# Initiating individual elements of the ML_ITEMS
encoder  = loaded_toolkit["encoder"]
# scaler = loaded_toolkit["scaler"]
model = loaded_toolkit["model"]

# Function to get date features from the inputs
def getDateFeatures(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['is_weekend'] = np.where(df['day_of_week'] > 4, 1, 0)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['date'].dt.is_year_end.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    df = df.drop(columns = "date")
    return df



# print(type(encoder))

# Defining the base containers   / main sections of the app
header = st.container()
dataset = st.container()
features_and_output = st.container()

form = st.form(key="information", clear_on_submit=True)

# Structuring the header section
with header:
    #header.write("Sales Prediction")
    # Icon for the page
    image = Image.open("/Users/Bernard Ampomah/Desktop/BAP Accelerator Projects/Projects/LP 2 Time Series/groceries.jpeg")
    st.image(image, width = 600)

# Creating elements of the sidebat
st.sidebar.header("This app predicts the sales of the Corporation Favorita grocery store")
check =st.sidebar.checkbox("Click here for column discription")
if check:
    st.sidebar.markdown(""" 
                    - **store_nbr** identifies the store at which the products are sold.
                    - **family** identifies the type of product sold.
                    - **sales** this is the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units(1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
                    - **onpromotion** gives the total number of items in a product family that were being promoted at a store at a given date.
                    - **sales_date** is the date on which a transaction / sale was made
                    - **city** is the city in which the store is located
                    - **state** is the state in which the store is located
                    - **store_type** is the type of store, based on Corporation Favorita's own type system
                    - **oil_price** is the daily oil price
                    - **holiday_type** indicates whether the day was a holiday, event day, or a workday
                    - **locale** indicates whether the holiday was local, national or regional.
                    - **transferred** indicates whether the day was a transferred holiday or not.
                    """)
# Structuring the dataset section
with dataset:
       dataset.markdown("**This is the dataset of Corporation Favorita**")
       check = dataset.checkbox("Preview the dataset")
       if check:
            dataset.write(NewTrain.head())
       dataset.write("View sidebar for column discription")
     
# List of expected variables

expected_inputs = ["date",  "store_nbr", "family", "onpromotion", "city", 'type', 'locale' 'is_holiday', 'oil_price', 'season']

# List of features to encode
categoricals = ["family", "city", "type", "locale"]

# List of features to scale
# cols_to_scale = ['dcoilwtico']

with features_and_output:
    features_and_output.subheader("Provide your Inputs")
    features_and_output.write("This section captures your inputs for the sales predictions")

    cols1, cols2, cols3, cols4  = features_and_output.columns(4)


# Forms to retrieve input
with form:
    cols1 = st.columns((1, 1))
    date = cols1[0].date_input('select sales date', min_value = NewTrain['date'].min())
    Location = cols1[1].selectbox('Please select store location', options = list(NewTrain['city'].unique()))
    
    # Second row
    cols2 = st.columns((1, 1))
    Store_number = cols2[0].select_slider('select store number', 1,54 )
    Product_family = cols2[1].selectbox('What is the product family', options = list(NewTrain['family'].unique()))
    
    # Third row
    cols3 = st.columns((1, 1))
    Oil_price = cols3[0].number_input('Input Current fuel price')
    # On_promo = cols3[1].select_slider('Is the item on promotion', ['Yes', 'No'])
    on_promo = cols3[1].slider("Select number of items on promo:",min_value =0, max_value = 742,step =1)
    

    # Forth row
    cols4 = st.columns((1, 1))
    if cols4.checkbox("Is it a holiday? (Check if holiday)"):
            print(True)
            Type = cols4.selectbox("Holiday type:", options=(NewTrain["type"].unique()))
            locale = cols4.selectbox("Locale:", options=(NewTrain["locale"].unique()))
            # transferred = col2.radio('Was the holiday transferred',
            # ('True','False'))
            # print((NewTrain["type"].unique()))
    else:
        Type= "Work Day"
        locale = "National"
        # transferred = 'False'

##############################################################################################
    # with form:
    #     date = col1.date_input("Select a date:", min_value= NewTrain["date"].min())
    #     family =col1.selectbox("Family of items:",options=(list( NewTrain['family'].unique())))
    #     city = col1.selectbox("choose city:",options =(NewTrain['city'].unique()))
    #     store_nbr = col1.selectbox("store number:",options =(NewTrain['store_nbr'].unique()))
    #     # cluster = col1.selectbox("cluster:",options =(NewTrain['cluster'].unique()))
    #     Oil_price = col1.number_input('input current fuel price')
    #     onpromotion = col2.slider("Select number of items on promo:",min_value =0, max_value = 742,step =1)
    #     if col2.checkbox("Is it a holiday? (Check if holiday)"):
    #         print(True)
    #         type_y = col2.selectbox("Holiday type:", options=(NewTrain["type_y"].unique()))
    #         locale = col2.selectbox("Locale:", options=(NewTrain["locale"].unique()))
    #         transferred = col2.radio('Was the holiday transferred',
    #         ('True','False'))
    #         #print((train_data["type_y"].unique()))
    #     else:
    #         type_y = "Work Day"
    #         locale = "National"
    #         # transferred = 'False'
##############################################################################################       
       
       
        # Submit button
        submitted = form.form_submit_button(label= "Make Prediction")
        # transactions = col2.slider("Select the number transactions for this day:",min_value =0, max_value = 8359,step = 100)
    
    
    if submitted:
        st.success('All Done!', icon="✅")  
        # Inputs formatting
        input_dict = {
            "date": [date],
            "family": [Product_family],
            "store_nbr": [Store_number],
            # "cluster": [cluster],
            "city": [Location],
            "onpromotion": [on_promo],
            "oil_price": [Oil_price],
            "type_y": [Type],
            "locale": [locale],
           
            # "transferred":[transferred],
            # "transactions" :[transactions]
        }

        # Converting the input into a dataframe
        input_data = pd.DataFrame.from_dict(input_dict)
        input_df = input_data.copy()
        
        # Converting data types into required types
        input_data["date"] = pd.to_datetime(input_data["date"]).dt.date
        # input_data[cols_to_scale] = input_data[cols_to_scale].apply(int)
        
        # Getting date features
        processed_df = getDateFeatures(input_data, "date")

        processed_df['year'] = pd.to_datetime(processed_df['date']).dt.year
        processed_df['month'] = pd.to_datetime(processed_df['date']).dt.month
        processed_df['week'] = pd.to_datetime(processed_df['date']).dt.week
        processed_df['day'] = pd.to_datetime(processed_df['date']).dt.day
        processed_df.drop(columns=["date"], inplace= True)
       
        # Scaling the columns
        # processed_df[cols_to_scale] = mscaler.transform(processed_df[cols_to_scale])

        # Encoding the categoricals
        # print(categoricals)
        # print(encoder.feature_names_in_)
        #print(encoded_categoricals.get_feature_names_out().tolist())
        
        encoded_categoricals = encoder.transform(input_data[categoricals])
        encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = encoder.get_feature_names_out().tolist())
        processed_df = processed_df.join(encoded_categoricals)
        processed_df.drop(columns=categoricals, inplace=True)
        processed_df.rename(columns= lambda x: re.sub("[^A-Za-z0-9_]+", "", x), inplace= True)
        processed_df["transferred"] = np.where(processed_df["transferred"] == "True", 0, 1)
        
        # Making the predictions        
        dt_pred = model.predict(processed_df)
        processed_df["sales"] = dt_pred
        input_df["sales"] = dt_pred
        display = dt_pred[0]

        # concatinating predictions
        st.session_state["results"].append(input_df)
        result = pd.concat(st.session_state["results"])


        # Displaying prediction results
        st.success(f"**Predicted sales**: USD {display}")

       # Expander to display previous predictions
        previous_output = st.expander("**Review previous predictions**")
        previous_output.dataframe(result, use_container_width= True)
    
    
# ----- Defining and structuring the footer
footer = st.expander("**Additional Information**")
with footer:
    footer.markdown("""
                    - You may access the repository in which the model was built [here](https://github.com/MavisAJ/Store-sales-prediction-Regression___TimeSeries-Analysis-.git).
                    - This is my first attempt at a Streamlit project so I would love to hear your criticisms.
                    - You may also connect with me [here](https://kodoi-oj.github.io/).
                    - *KME*
                    """)