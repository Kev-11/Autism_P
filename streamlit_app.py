import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the encoder
with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Load the training data for feature information
train_df = pd.read_csv('datasets/train.csv')

# Preprocessing function
def preprocess_data(df, encoders):
    # Handle missing values in ethnicity and relation columns
    df['ethnicity'] = df['ethnicity'].replace({"?":"Others","others":"Others"})
    df['relation'] = df['relation'].replace({
        "?":"Others",
        "Relative" : "Others",
        "Parent" : "Others",
        "Health care professional" : "Others"
    })

    # Label Encoding
    object_columns = df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        if column in encoders:
            try:
                df[column] = encoders[column].transform(df[column])
            except ValueError:
                # Handle unseen labels by replacing with the most frequent label
                most_frequent_label = train_df[column].mode()[0]
                df[column] = [most_frequent_label] * len(df)
                df[column] = encoders[column].transform(df[column])
    return df

# Streamlit app
def main():
    st.title('Autism Spectrum Disorder Prediction')

    # Create input fields for features
    a1_score = st.slider('A1_Score', 0, 1, 0)
    a2_score = st.slider('A2_Score', 0, 1, 0)
    a3_score = st.slider('A3_Score', 0, 1, 0)
    a4_score = st.slider('A4_Score', 0, 1, 0)
    a5_score = st.slider('A5_Score', 0, 1, 0)
    a6_score = st.slider('A6_Score', 0, 1, 0)
    a7_score = st.slider('A7_Score', 0, 1, 0)
    a8_score = st.slider('A8_Score', 0, 1, 0)
    a9_score = st.slider('A9_Score', 0, 1, 0)
    a10_score = st.slider('A10_Score', 0, 1, 0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', 0, 100, 25)
    ethnicity = st.selectbox('Ethnicity', train_df['ethnicity'].unique())
    jaundice = st.selectbox('Jaundice', ['Yes', 'No'])
    austim = st.selectbox('Family member with ASD', ['Yes', 'No'])
    contry_of_res = st.selectbox('Country of Residence', train_df['contry_of_res'].unique())
    used_app_before = st.selectbox('Used App Before', ['Yes', 'No'])
    relation = st.selectbox('Relation', train_df['relation'].unique())
    result = st.number_input('Result', 0, 10, 5)

    # Create a dataframe from the input values with the same column order as the training data
    feature_columns = [col for col in train_df.columns if col not in ['ID', 'age_desc', 'Class/ASD']]

    # Create a dataframe from the input values
    input_data_dict = {
        'A1_Score': [a1_score],
        'A2_Score': [a2_score],
        'A3_Score': [a3_score],
        'A4_Score': [a4_score],
        'A5_Score': [a5_score],
        'A6_Score': [a6_score],
        'A7_Score': [a7_score],
        'A8_Score': [a8_score],
        'A9_Score': [a9_score],
        'A10_Score': [a10_score],
        'gender': [gender],
        'age': [age],
        'ethnicity': [ethnicity],
        'jaundice': [jaundice],
        'austim': [austim],
        'contry_of_res': [contry_of_res],
        'used_app_before': [used_app_before],
        'relation': [relation],
        'result': [result]
    }
    input_data = pd.DataFrame(input_data_dict)

    # Preprocess the input data
    input_data = preprocess_data(input_data, encoders)

    # Ensure the column order matches the training data
    input_data = input_data[feature_columns]

    # Make prediction
    if st.button('Predict'):
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.write('The person is not predicted to have Autism Spectrum Disorder.')
        else:
            st.write('The person is predicted to have Autism Spectrum Disorder.')

if __name__ == '__main__':
    main()
