# Autism Spectrum Disorder Prediction

This project is a Streamlit web application that predicts whether a person has Autism Spectrum Disorder (ASD) based on a set of input features.

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository_url]
    ```
2.  Navigate to the project directory:

    ```bash
    cd Autism_P
    ```
3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit app:

    ```bash
    streamlit run streamlit_app.py
    ```
2.  Open the app in your browser at the address displayed in the terminal (usually `http://localhost:8501`).
3.  Enter the required information in the input fields.
4.  Click the "Predict" button to get the prediction.

## Deployment on Streamlit

1.  Create a GitHub repository for your project.
2.  Push your local repository to GitHub.
3.  Create a Streamlit account (if you don't have one already) at [https://streamlit.io/](https://streamlit.io/).
4.  Click on the "New app" button.
5.  Select the GitHub repository you created.
6.  Specify the branch, the main script file (`streamlit_app.py`), and any necessary dependencies.
7.  Click the "Deploy!" button.

## Images

Add project images here to showcase the app's interface and functionality.

## Model

The model used for prediction is a Random Forest Classifier trained on a dataset of individuals with and without ASD.

## Encoders

Label encoders are used to transform categorical features into numerical values.
