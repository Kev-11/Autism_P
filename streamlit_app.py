import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="ASD Prediction Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black and white theme
st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 2px solid #FFFFFF;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FFFFFF;
    }
    
    /* Input widgets */
    .stSlider > div > div > div {
        background-color: #FFFFFF;
    }
    
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
    }
    
    .stNumberInput > div > div {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #FFFFFF;
        color: #000000;
        border: 2px solid #FFFFFF;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #000000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
        transform: scale(1.02);
    }
    
    /* Quick Actions Section in Sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #000000;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #FFFFFF;
        color: #000000;
        border: 2px solid #FFFFFF;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #FFFFFF;
        color: #FFFFFF;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000000;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
        border-radius: 4px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Cards */
    .card {
        background-color: #1a1a1a;
        border: 2px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        transition: transform 0.2s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(255, 255, 255, 0.2);
    }
    
    /* Progress indication */
    .progress-text {
        color: #FFFFFF;
        font-size: 14px;
        margin-top: 10px;
    }
    
    /* Result boxes */
    .result-box {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: 700;
        margin: 20px 0;
        border: 3px solid;
        animation: fadeIn 0.5s;
    }
    
    .positive-result {
        background-color: #1a1a1a;
        border-color: #FF0000;
        color: #FFFFFF;
    }
    
    .negative-result {
        background-color: #1a1a1a;
        border-color: #00FF00;
        color: #FFFFFF;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Divider */
    hr {
        border: 1px solid #FFFFFF;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

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
    # Header with custom styling
    st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>🧠 Autism Spectrum Disorder Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #CCCCCC; font-size: 18px;'>Interactive Assessment Tool</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Initialize session state for active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Initialize session state for storing tab data
    if 'tab_data' not in st.session_state:
        st.session_state.tab_data = {}
    
    # Sidebar for information and settings
    with st.sidebar:
        st.markdown("### 📋 About This Tool")
        st.markdown("""
        This tool uses machine learning to predict the likelihood of Autism Spectrum Disorder (ASD) based on behavioral assessment scores and demographic information.
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Actions")
        if st.button("🔄 Reset All Fields"):
            st.rerun()
        
        st.markdown("---")
        
        # Assessment questions info in expander
        with st.expander("ℹ️ Assessment Questions Guide"):
            st.markdown("""
            **A1-A10 Scores** represent responses to 10 behavioral assessment questions:
            - **Unchecked** = No / Not Present
            - **Checked** = Yes / Present
            
            Each question relates to specific behavioral traits commonly associated with ASD. Simply check the boxes that apply to the individual being assessed.
            """)
        
        st.markdown("---")
        
        # Statistics display
        st.markdown("### 📊 Current Session Stats")
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        st.metric("Predictions Made", st.session_state.prediction_count)

    # Create navigation bar
    st.markdown("### Navigation")
    tab_names = ["📝 Assessment Scores", "👤 Personal Information", "🔬 Prediction"]
    
    cols = st.columns(3)
    for idx, tab_name in enumerate(tab_names):
        with cols[idx]:
            button_style = "primary" if st.session_state.active_tab == idx else "secondary"
            if st.button(tab_name, key=f"tab_button_{idx}", use_container_width=True, type=button_style):
                st.session_state.active_tab = idx
                st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Tab 1: Assessment Scores
    if st.session_state.active_tab == 0:
        st.markdown("### Behavioral Assessment Scores")
        st.markdown("*Select the checkboxes if true for each question*")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Questions 1-5")
            a1_score = 1 if st.checkbox('A1: Does your child look at you when you call their name?', value=False) else 0
            a2_score = 1 if st.checkbox('A2: How easy is it for you to get eye contact with your child?', value=False, help="Check if eye contact is easy") else 0
            a3_score = 1 if st.checkbox('A3: Does your child point to indicate they want something?', value=False) else 0
            a4_score = 1 if st.checkbox('A4: Does your child point to share interest with you?', value=False) else 0
            a5_score = 1 if st.checkbox('A5: Does your child pretend in play?', value=False) else 0
        
        with col2:
            st.markdown("#### Questions 6-10")
            a6_score = 1 if st.checkbox('A6: Does your child follow where you are looking?', value=False) else 0
            a7_score = 1 if st.checkbox('A7: If you are upset, does your child show signs of comforting?', value=False) else 0
            a8_score = 1 if st.checkbox('A8: Would you describe your child\'s first words as unusual?', value=False) else 0
            a9_score = 1 if st.checkbox('A9: Does your child use simple gestures?', value=False) else 0
            a10_score = 1 if st.checkbox('A10: Does your child stare at nothing with no apparent purpose?', value=False) else 0
        
        # Display total score
        total_score = sum([a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score])
        st.markdown(f"<div class='card'><h3 style='text-align: center;'>Total Assessment Score: {total_score}/10</h3></div>", unsafe_allow_html=True)
        
        # Store scores in session state
        st.session_state.tab_data['a1_score'] = a1_score
        st.session_state.tab_data['a2_score'] = a2_score
        st.session_state.tab_data['a3_score'] = a3_score
        st.session_state.tab_data['a4_score'] = a4_score
        st.session_state.tab_data['a5_score'] = a5_score
        st.session_state.tab_data['a6_score'] = a6_score
        st.session_state.tab_data['a7_score'] = a7_score
        st.session_state.tab_data['a8_score'] = a8_score
        st.session_state.tab_data['a9_score'] = a9_score
        st.session_state.tab_data['a10_score'] = a10_score
        st.session_state.tab_data['total_score'] = total_score
        
        # Navigation buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col3:
            if st.button("Next: Personal Info →", key="nav_to_tab2", use_container_width=True):
                st.session_state.active_tab = 1
                st.rerun()
    
    # Tab 2: Personal Information
    elif st.session_state.active_tab == 1:
        st.markdown("### Personal & Medical Information")
        
        # Get total_score from session state
        total_score = st.session_state.tab_data.get('total_score', 0)
        
        # Display assessment score from previous tab
        st.markdown(f"<div class='card'><h4 style='text-align: center;'>Assessment Score: <span style='color: #FFFFFF; font-size: 24px;'>{total_score}/10</span></h4></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demographics")
            gender = st.radio('Gender', ['Male', 'Female'], horizontal=True)
            age = st.number_input('Age', 0, 100, 25, help="Enter the age of the individual")
            ethnicity_options = sorted([e for e in train_df['ethnicity'].unique() if e != '?'])
            ethnicity = st.selectbox('Ethnicity', ethnicity_options, help="Select ethnicity")
            contry_of_res = st.selectbox('Country of Residence', sorted(train_df['contry_of_res'].unique()), help="Select country")
        
        with col2:
            st.markdown("#### Medical & Family History")
            jaundice = st.radio('Born with Jaundice?', ['Yes', 'No'], horizontal=True)
            austim = st.radio('Family member with ASD?', ['Yes', 'No'], horizontal=True)
            used_app_before = st.radio('Used screening app before?', ['Yes', 'No'], horizontal=True)
            relation_options = sorted([r for r in train_df['relation'].unique() if r != '?'])
            relation = st.selectbox('Who is completing this test?', relation_options, help="Relation to the individual")
        
        # Use total_score from assessment tab directly
        result = total_score
        
        # Store personal info in session state
        st.session_state.tab_data['gender'] = gender
        st.session_state.tab_data['age'] = age
        st.session_state.tab_data['ethnicity'] = ethnicity
        st.session_state.tab_data['contry_of_res'] = contry_of_res
        st.session_state.tab_data['jaundice'] = jaundice
        st.session_state.tab_data['austim'] = austim
        st.session_state.tab_data['used_app_before'] = used_app_before
        st.session_state.tab_data['relation'] = relation
        st.session_state.tab_data['result'] = result
        
        # Navigation buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Back: Assessment", key="nav_to_tab1", use_container_width=True):
                st.session_state.active_tab = 0
                st.rerun()
        with col3:
            if st.button("Next: Prediction →", key="nav_to_tab3", use_container_width=True):
                st.session_state.active_tab = 2
                st.rerun()
    
    # Tab 3: Prediction
    elif st.session_state.active_tab == 2:
        # Get all values from session state
        a1_score = st.session_state.tab_data.get('a1_score', 0)
        a2_score = st.session_state.tab_data.get('a2_score', 0)
        a3_score = st.session_state.tab_data.get('a3_score', 0)
        a4_score = st.session_state.tab_data.get('a4_score', 0)
        a5_score = st.session_state.tab_data.get('a5_score', 0)
        a6_score = st.session_state.tab_data.get('a6_score', 0)
        a7_score = st.session_state.tab_data.get('a7_score', 0)
        a8_score = st.session_state.tab_data.get('a8_score', 0)
        a9_score = st.session_state.tab_data.get('a9_score', 0)
        a10_score = st.session_state.tab_data.get('a10_score', 0)
        total_score = st.session_state.tab_data.get('total_score', 0)
        
        gender = st.session_state.tab_data.get('gender', 'Male')
        age = st.session_state.tab_data.get('age', 25)
        ethnicity = st.session_state.tab_data.get('ethnicity', 'Others')
        contry_of_res = st.session_state.tab_data.get('contry_of_res', 'United States')
        jaundice = st.session_state.tab_data.get('jaundice', 'No')
        austim = st.session_state.tab_data.get('austim', 'No')
        used_app_before = st.session_state.tab_data.get('used_app_before', 'No')
        relation = st.session_state.tab_data.get('relation', 'Self')
        result = st.session_state.tab_data.get('result', 0)
        
        st.markdown("### 🔬 Generate Prediction")
        st.markdown("Review your inputs and click the button below to generate a prediction.")
        
        # Display summary in an interactive card
        with st.expander("📊 View Input Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Assessment Scores:**")
                st.write(f"Total Score: {total_score}/10")
                st.write(f"Individual Scores: {a1_score}, {a2_score}, {a3_score}, {a4_score}, {a5_score}, {a6_score}, {a7_score}, {a8_score}, {a9_score}, {a10_score}")
            
            with col2:
                st.markdown("**Demographics:**")
                st.write(f"Gender: {gender}")
                st.write(f"Age: {age}")
                st.write(f"Ethnicity: {ethnicity}")
            
            with col3:
                st.markdown("**Medical History:**")
                st.write(f"Jaundice: {jaundice}")
                st.write(f"Family ASD: {austim}")
                st.write(f"Previous Screening: {used_app_before}")

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

        # Make prediction with interactive button
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button('🔮 GENERATE PREDICTION', use_container_width=True)
        
        if predict_button:
            # Show loading animation
            with st.spinner('Analyzing data...'):
                import time
                time.sleep(1)  # Simulate processing time for better UX
                
                prediction = model.predict(input_data)
                st.session_state.prediction_count += 1
                
                # Display prediction with styled result box
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction[0] == 0:
                    st.markdown("""
                        <div class='result-box negative-result'>
                            ✅ PREDICTION: LOW LIKELIHOOD<br>
                            <span style='font-size: 16px; font-weight: normal;'>
                            The individual is NOT predicted to have Autism Spectrum Disorder based on the provided information.
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("This assessment suggests a low likelihood of ASD. However, this is not a diagnostic tool. Please consult healthcare professionals for proper evaluation.")
                    
                else:
                    st.markdown("""
                        <div class='result-box positive-result'>
                            ⚠️ PREDICTION: HIGH LIKELIHOOD<br>
                            <span style='font-size: 16px; font-weight: normal;'>
                            The individual IS predicted to have Autism Spectrum Disorder based on the provided information.
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning("This assessment suggests a higher likelihood of ASD. Please consult qualified healthcare professionals for comprehensive evaluation and diagnosis.")
                
                # Additional recommendations
                with st.expander("📋 Next Steps & Recommendations"):
                    st.markdown("""
                    ### Important Notes:
                    - This tool provides predictions based on machine learning models and should NOT be used as a sole diagnostic tool.
                    - A formal diagnosis requires comprehensive evaluation by qualified healthcare professionals.
                    - Early intervention can be beneficial, so consult with specialists if concerns persist.
                    
                    ### Recommended Actions:
                    1. **Consult a Healthcare Professional**: Schedule an appointment with a pediatrician or psychologist specializing in developmental disorders.
                    2. **Comprehensive Evaluation**: Seek a full diagnostic assessment if recommended by professionals.
                    3. **Document Observations**: Keep notes on behavioral patterns and concerns to share with healthcare providers.
                    4. **Support Resources**: Research local ASD support groups and resources.
                    """)
                
                # Option to save or export results
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 View Detailed Report", use_container_width=True):
                        st.info("Detailed reporting feature - coming soon!")
                
                with col2:
                    if st.button("🔄 Start New Assessment", use_container_width=True):
                        st.rerun()
        
        # Navigation button for this tab (when no prediction made yet)
        if not predict_button:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("← Back: Personal Info", key="nav_back_to_tab2", use_container_width=True):
                    st.session_state.active_tab = 1
                    st.rerun()

if __name__ == '__main__':
    main()
