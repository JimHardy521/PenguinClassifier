import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# JimHardy-DataScience:
Penguin Prediction App

This app predicts Penguin Species.

Tutorial following from: https://www.youtube.com/watch?v=Eai1jaZrRDs


""")

st.sidebar.header('Select Feature Attributes:')


# Collects user input features into dataframe
def user_input_features():
    island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
    sex = st.sidebar.selectbox('Sex',('male','female'))
    bill_length_mm = st.sidebar.slider('Bill length (mm)', 30.0,60.0,45.0)
    bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 10.0,25.0,17.5)
    flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 150.0,250.0,200.0)
    body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4500.0)
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')
input_df1 = input_df.rename(columns= {'island':'Island','bill_length_mm':'Bill Length','bill_depth_mm':'Bill Depth','flipper_length_mm':'Flipper Length','body_mass_g':'Body Mass','sex':'Sex'})
st.write(input_df1)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))


# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

proba_df = pd.DataFrame(prediction_proba)
column_names = ['Adelie', 'Chinstrap', 'Gentoo']
proba_df.columns = column_names

st.subheader('Prediction Probability')
st.write(proba_df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])


