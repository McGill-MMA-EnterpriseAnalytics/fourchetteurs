import streamlit as st
import pandas as pd
import requests

# Function to send requests to FastAPI
def send_request(data):
    url = 'http://localhost:8000/predict/'  # Adjust the URL based on your setup
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data)
    return response.json()

st.title('Bank Marketing Prediction Interface')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Assuming the CSV has the same format required by the model
    if st.button('Predict'):
        # Convert DataFrame to JSON
        json_data = {"features": data.values.tolist()}
        response = send_request(json_data)

        if 'prediction' in response:
            # Add predictions to the DataFrame
            data['Prediction'] = response['prediction']

            st.write('Predictions:')
            st.write(data)

            # Convert DataFrame to CSV and create download link
            csv = data.to_csv(index=False)
            st.download_button(label="Download predictions as CSV",
                               data=csv,
                               file_name='predictions.csv',
                               mime='text/csv')
        else:
            st.error('Error in prediction: {}'.format(response['error']))

