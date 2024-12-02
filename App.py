import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model = joblib.load('loan_approval_model.pkl')

# Load dataset (if needed for initial analysis)
data = pd.read_csv("LoanApprovalPrediction.csv")

# Drop columns that are not used during prediction
data = data.drop(['Loan_ID', 'Loan_Status'], axis=1)

# Set page title and icon
st.set_page_config(page_title="Loan Approval Prediction", page_icon=":guardsman:", layout="wide")

# Add a header and background image
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://your-image-url.com");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
    }
    </style>
""", unsafe_allow_html=True)

# Title with an image
st.title("Loan Approval Prediction")
st.markdown("<h3 style=color: white;'>Predict if your loan will be approved based on the inputs.</h3>", unsafe_allow_html=True)

# You can add an image using st.image() at the top of the page or near the title
# Center the image using HTML within st.markdown
st.image("loan.png", width=200)


# Create a form layout for inputs
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        # Use slider for numeric inputs to make them more flexible and user-friendly
        applicant_income = st.slider('Applicant Income', min_value=0, max_value=100000, step=500)
        coapplicant_income = st.slider('Coapplicant Income', min_value=0, max_value=100000, step=500)
        loan_amount = st.slider('Loan Amount', min_value=0, max_value=100000, step=1000)
        loan_term = st.slider('Loan Term (months)', min_value=0, max_value=360, step=12)

        gender = st.selectbox('Gender', ['Male', 'Female'])
        married = st.selectbox('Marital Status', ['Married', 'Not Married'])
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'])

    with col2:
        self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
        credit_history = st.selectbox('Credit History', [1, 0])  # Assuming 1 for good, 0 for bad
        property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
        dependents = st.selectbox('Dependents', [0, 1, 2, 3])  # Adjust this based on your needs

    # Submit button for form
    submit_button = st.form_submit_button(label='Predict Loan Approval')

# When the form is submitted
if submit_button:
    # Label encoding for the user inputs
    label_encoder = LabelEncoder()
    gender = label_encoder.fit_transform([gender])[0]
    married = label_encoder.fit_transform([married])[0]
    education = label_encoder.fit_transform([education])[0]
    self_employed = label_encoder.fit_transform([self_employed])[0]
    property_area = label_encoder.fit_transform([property_area])[0]

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[gender, married, education, self_employed, applicant_income,
                                coapplicant_income, loan_amount, loan_term, credit_history, property_area, dependents]],
                              columns=['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome',
                                       'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                                       'Property_Area', 'Dependents'])

    # Ensure that the input data has the same columns as the training data
    input_data = input_data[data.columns]  # Align columns with the model's training data

    # Make prediction
    prediction = model.predict(input_data)

    # Display the result with enhanced styling
   # Display the result with enhanced styling
if prediction[0] == 1:
    st.markdown('<h3 style="color: green; text-align: center;">Loan Approved!</h3>', unsafe_allow_html=True)
else:
    st.markdown('<h3 style="color: red; text-align: center;">Loan Not Approved</h3>', unsafe_allow_html=True)
