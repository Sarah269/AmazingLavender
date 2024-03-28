# Respiratory Illness Classification App
# Machine Learning Classification Model
# Pydataset Respiratory

# Load Libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# import dataset from pydataset
from pydataset import data

# Use the respiratory dataset
# Respiratory Illness Data
df = data('respiratory')

# Convert categorical features treat and sex to numeric values

from sklearn.preprocessing import LabelEncoder

df[['treat','sex']] = df[['treat', 'sex']].apply(LabelEncoder().fit_transform)

# Encoding
# * Treat:  P = 1, A = 0
# * Sex: M= 1, F = 0

# Separate the features for the target 

#Features
X = df.drop(['outcome'], axis=1)
#X.head()

# Target
y = df['outcome']
#y.head()

# Address imbalance between outcome values

from imblearn.over_sampling import RandomOverSampler

#Oversampling & fit
ros = RandomOverSampler()
X_res,y_res = ros.fit_resample(X,y)

# Write Title for Streamlit App
st.write("""
# Simple Prediction App 

This app predicts Respiratory Illness 
""")
# Write Streamlit Sidebar Title
st.sidebar.header('User Input Parameters')

# Define User Input Feature Function
def user_input_features():
   center = st.sidebar.slider('Center',1,2,1)
   id = st.sidebar.slider('ID',1,56,28) 
   treat = st.sidebar.slider('Treat (P = 1, A = 0)',0,1,0)
   sex = st.sidebar.slider('Sex (M = 1, F= 0)',0,1,0) 
   age = st.sidebar.slider('Age',X_res.age.min(), X_res.age.max(),33)
   baseline = st.sidebar.slider('Baseline',0,1,0)
   visit = st.sidebar.slider('Visit',1,4,2)
    
   data = {'center': center,
           'id': id,
           'treat': treat,
           'sex': sex,
           'age': age,
           'baseline': baseline,
           'visit': visit
          }          
   features = pd.DataFrame(data, index=[0])
   return features

# Capture user selected features from sidebar
df_input = user_input_features()

# Write User Input Parameters
st.subheader('User Input Parameters')
st.write(df_input)
st.write("___")

# Build model with user selected parameters
rf_model = RandomForestClassifier(random_state = 42)

# Fit model
rf_model.fit(X_res,y_res)

# Predict with model
prediction = rf_model.predict(df_input)
prediction_proba = rf_model.predict_proba(df_input)

# List possible outcomes with labels
st.subheader('Class labels and corresponding index')

labels = np.array(['No Respiratory Illness','Respiratory Illness'])
st.write(pd.DataFrame(labels))

# List Prediction outcome
st.subheader('Prediction of Illness')
st.write(labels[prediction])

# List Prediction Probability
st.subheader('Prediction Probability')
st.write(prediction_proba)

