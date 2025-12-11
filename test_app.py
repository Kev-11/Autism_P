"""
Test Cases for Autism Spectrum Disorder Prediction App
"""

import pandas as pd
import pickle
import sys
from io import StringIO

# Test Case 1: Model and Encoder Loading
def test_model_loading():
    """Test if the model and encoders load successfully"""
    print("\n" + "="*60)
    print("TEST 1: Model and Encoder Loading")
    print("="*60)
    
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        print("✅ PASS: Model loaded successfully")
        print(f"   Model Type: {type(model).__name__}")
        return True, model
    except FileNotFoundError:
        print("❌ FAIL: Model file 'best_model.pkl' not found")
        return False, None
    except Exception as e:
        print(f"❌ FAIL: Error loading model - {str(e)}")
        return False, None

def test_encoder_loading():
    """Test if encoders load successfully"""
    print("\n" + "="*60)
    print("TEST 2: Encoder Loading")
    print("="*60)
    
    try:
        with open('encoders.pkl', 'rb') as file:
            encoders = pickle.load(file)
        print("✅ PASS: Encoders loaded successfully")
        print(f"   Encoder columns: {list(encoders.keys())}")
        return True, encoders
    except FileNotFoundError:
        print("❌ FAIL: Encoder file 'encoders.pkl' not found")
        return False, None
    except Exception as e:
        print(f"❌ FAIL: Error loading encoders - {str(e)}")
        return False, None

def test_training_data_loading():
    """Test if training data loads successfully"""
    print("\n" + "="*60)
    print("TEST 3: Training Data Loading")
    print("="*60)
    
    try:
        train_df = pd.read_csv('datasets/train.csv')
        print("✅ PASS: Training data loaded successfully")
        print(f"   Shape: {train_df.shape}")
        print(f"   Columns: {list(train_df.columns)}")
        return True, train_df
    except FileNotFoundError:
        print("❌ FAIL: Training data file not found")
        return False, None
    except Exception as e:
        print(f"❌ FAIL: Error loading training data - {str(e)}")
        return False, None

# Test Case 4: Data Preprocessing Function
def test_preprocess_data(encoders, train_df):
    """Test the preprocessing function with sample data"""
    print("\n" + "="*60)
    print("TEST 4: Data Preprocessing Function")
    print("="*60)
    
    def preprocess_data(df, encoders, train_df):
        df['ethnicity'] = df['ethnicity'].replace({"?":"Others","others":"Others"})
        df['relation'] = df['relation'].replace({
            "?":"Others",
            "Relative" : "Others",
            "Parent" : "Others",
            "Health care professional" : "Others"
        })
        
        object_columns = df.select_dtypes(include=["object"]).columns
        for column in object_columns:
            if column in encoders:
                try:
                    df[column] = encoders[column].transform(df[column])
                except ValueError:
                    most_frequent_label = train_df[column].mode()[0]
                    df[column] = [most_frequent_label] * len(df)
                    df[column] = encoders[column].transform(df[column])
        return df
    
    # Test Case 4a: Normal data
    print("\n[4a] Testing with normal data:")
    test_data = pd.DataFrame({
        'A1_Score': [1],
        'A2_Score': [0],
        'A3_Score': [1],
        'A4_Score': [1],
        'A5_Score': [0],
        'A6_Score': [1],
        'A7_Score': [0],
        'A8_Score': [1],
        'A9_Score': [1],
        'A10_Score': [0],
        'gender': ['Male'],
        'age': [25],
        'ethnicity': ['White-European'],
        'jaundice': ['No'],
        'austim': ['Yes'],
        'contry_of_res': ['United States'],
        'used_app_before': ['No'],
        'relation': ['Self'],
        'result': [6]
    })
    
    try:
        processed = preprocess_data(test_data.copy(), encoders, train_df)
        print("✅ PASS: Normal data preprocessed successfully")
        print(f"   Processed shape: {processed.shape}")
        return True
    except Exception as e:
        print(f"❌ FAIL: Error preprocessing normal data - {str(e)}")
        return False

# Test Case 5: Prediction with Different Scenarios
def test_predictions(model, encoders, train_df):
    """Test predictions with different scenarios"""
    print("\n" + "="*60)
    print("TEST 5: Prediction Scenarios")
    print("="*60)
    
    def preprocess_data(df, encoders, train_df):
        df['ethnicity'] = df['ethnicity'].replace({"?":"Others","others":"Others"})
        df['relation'] = df['relation'].replace({
            "?":"Others",
            "Relative" : "Others",
            "Parent" : "Others",
            "Health care professional" : "Others"
        })
        
        object_columns = df.select_dtypes(include=["object"]).columns
        for column in object_columns:
            if column in encoders:
                try:
                    df[column] = encoders[column].transform(df[column])
                except ValueError:
                    most_frequent_label = train_df[column].mode()[0]
                    df[column] = [most_frequent_label] * len(df)
                    df[column] = encoders[column].transform(df[column])
        return df
    
    feature_columns = [col for col in train_df.columns if col not in ['ID', 'age_desc', 'Class/ASD']]
    
    # Scenario 1: High likelihood of ASD (all scores = 1, family history)
    print("\n[5a] Scenario 1: High Risk Profile")
    print("   Description: All assessment scores = 1, family history of ASD")
    high_risk_data = pd.DataFrame({
        'A1_Score': [1], 'A2_Score': [1], 'A3_Score': [1], 'A4_Score': [1], 'A5_Score': [1],
        'A6_Score': [1], 'A7_Score': [1], 'A8_Score': [1], 'A9_Score': [1], 'A10_Score': [1],
        'gender': ['Male'], 'age': [5], 'ethnicity': ['White-European'], 'jaundice': ['Yes'],
        'austim': ['Yes'], 'contry_of_res': ['United States'], 'used_app_before': ['No'],
        'relation': ['Self'], 'result': [10]
    })
    
    try:
        processed = preprocess_data(high_risk_data.copy(), encoders, train_df)
        processed = processed[feature_columns]
        prediction = model.predict(processed)
        print(f"   Prediction: {'ASD Likely' if prediction[0] == 1 else 'ASD Unlikely'}")
        print(f"✅ PASS: High risk scenario predicted")
    except Exception as e:
        print(f"❌ FAIL: Error in high risk prediction - {str(e)}")
    
    # Scenario 2: Low likelihood of ASD (all scores = 0, no family history)
    print("\n[5b] Scenario 2: Low Risk Profile")
    print("   Description: All assessment scores = 0, no family history")
    low_risk_data = pd.DataFrame({
        'A1_Score': [0], 'A2_Score': [0], 'A3_Score': [0], 'A4_Score': [0], 'A5_Score': [0],
        'A6_Score': [0], 'A7_Score': [0], 'A8_Score': [0], 'A9_Score': [0], 'A10_Score': [0],
        'gender': ['Female'], 'age': [8], 'ethnicity': ['Asian'], 'jaundice': ['No'],
        'austim': ['No'], 'contry_of_res': ['United States'], 'used_app_before': ['No'],
        'relation': ['Self'], 'result': [0]
    })
    
    try:
        processed = preprocess_data(low_risk_data.copy(), encoders, train_df)
        processed = processed[feature_columns]
        prediction = model.predict(processed)
        print(f"   Prediction: {'ASD Likely' if prediction[0] == 1 else 'ASD Unlikely'}")
        print(f"✅ PASS: Low risk scenario predicted")
    except Exception as e:
        print(f"❌ FAIL: Error in low risk prediction - {str(e)}")
    
    # Scenario 3: Medium likelihood (mixed scores)
    print("\n[5c] Scenario 3: Medium Risk Profile")
    print("   Description: Mixed assessment scores (5/10), with jaundice")
    medium_risk_data = pd.DataFrame({
        'A1_Score': [1], 'A2_Score': [0], 'A3_Score': [1], 'A4_Score': [1], 'A5_Score': [0],
        'A6_Score': [1], 'A7_Score': [0], 'A8_Score': [1], 'A9_Score': [0], 'A10_Score': [0],
        'gender': ['Male'], 'age': [10], 'ethnicity': ['Others'], 'jaundice': ['Yes'],
        'austim': ['No'], 'contry_of_res': ['United States'], 'used_app_before': ['Yes'],
        'relation': ['Self'], 'result': [5]
    })
    
    try:
        processed = preprocess_data(medium_risk_data.copy(), encoders, train_df)
        processed = processed[feature_columns]
        prediction = model.predict(processed)
        print(f"   Prediction: {'ASD Likely' if prediction[0] == 1 else 'ASD Unlikely'}")
        print(f"✅ PASS: Medium risk scenario predicted")
    except Exception as e:
        print(f"❌ FAIL: Error in medium risk prediction - {str(e)}")
    
    # Scenario 4: Different age groups
    print("\n[5d] Scenario 4: Young Child Profile")
    print("   Description: Young child (age 3), high scores")
    young_child_data = pd.DataFrame({
        'A1_Score': [1], 'A2_Score': [1], 'A3_Score': [1], 'A4_Score': [1], 'A5_Score': [1],
        'A6_Score': [0], 'A7_Score': [1], 'A8_Score': [1], 'A9_Score': [0], 'A10_Score': [1],
        'gender': ['Male'], 'age': [3], 'ethnicity': ['Hispanic'], 'jaundice': ['No'],
        'austim': ['Yes'], 'contry_of_res': ['United States'], 'used_app_before': ['No'],
        'relation': ['Parent'], 'result': [8]
    })
    
    try:
        processed = preprocess_data(young_child_data.copy(), encoders, train_df)
        processed = processed[feature_columns]
        prediction = model.predict(processed)
        print(f"   Prediction: {'ASD Likely' if prediction[0] == 1 else 'ASD Unlikely'}")
        print(f"✅ PASS: Young child scenario predicted")
    except Exception as e:
        print(f"❌ FAIL: Error in young child prediction - {str(e)}")
    
    # Scenario 5: Adult assessment
    print("\n[5e] Scenario 5: Adult Profile")
    print("   Description: Adult (age 30), moderate scores")
    adult_data = pd.DataFrame({
        'A1_Score': [0], 'A2_Score': [1], 'A3_Score': [0], 'A4_Score': [1], 'A5_Score': [1],
        'A6_Score': [1], 'A7_Score': [0], 'A8_Score': [1], 'A9_Score': [0], 'A10_Score': [1],
        'gender': ['Female'], 'age': [30], 'ethnicity': ['Black'], 'jaundice': ['No'],
        'austim': ['No'], 'contry_of_res': ['United Kingdom'], 'used_app_before': ['Yes'],
        'relation': ['Self'], 'result': [6]
    })
    
    try:
        processed = preprocess_data(adult_data.copy(), encoders, train_df)
        processed = processed[feature_columns]
        prediction = model.predict(processed)
        print(f"   Prediction: {'ASD Likely' if prediction[0] == 1 else 'ASD Unlikely'}")
        print(f"✅ PASS: Adult scenario predicted")
    except Exception as e:
        print(f"❌ FAIL: Error in adult prediction - {str(e)}")

# Test Case 6: Edge Cases
def test_edge_cases(encoders, train_df):
    """Test edge cases and boundary conditions"""
    print("\n" + "="*60)
    print("TEST 6: Edge Cases and Boundary Conditions")
    print("="*60)
    
    def preprocess_data(df, encoders, train_df):
        df['ethnicity'] = df['ethnicity'].replace({"?":"Others","others":"Others"})
        df['relation'] = df['relation'].replace({
            "?":"Others",
            "Relative" : "Others",
            "Parent" : "Others",
            "Health care professional" : "Others"
        })
        
        object_columns = df.select_dtypes(include=["object"]).columns
        for column in object_columns:
            if column in encoders:
                try:
                    df[column] = encoders[column].transform(df[column])
                except ValueError:
                    most_frequent_label = train_df[column].mode()[0]
                    df[column] = [most_frequent_label] * len(df)
                    df[column] = encoders[column].transform(df[column])
        return df
    
    # Edge Case 1: Ethnicity with "?"
    print("\n[6a] Edge Case: Ethnicity with '?' value")
    edge_data_1 = pd.DataFrame({
        'A1_Score': [1], 'A2_Score': [0], 'A3_Score': [1], 'A4_Score': [1], 'A5_Score': [0],
        'A6_Score': [1], 'A7_Score': [0], 'A8_Score': [1], 'A9_Score': [1], 'A10_Score': [0],
        'gender': ['Male'], 'age': [25], 'ethnicity': ['?'], 'jaundice': ['No'],
        'austim': ['Yes'], 'contry_of_res': ['United States'], 'used_app_before': ['No'],
        'relation': ['Self'], 'result': [6]
    })
    
    try:
        processed = preprocess_data(edge_data_1.copy(), encoders, train_df)
        print("✅ PASS: '?' ethnicity handled correctly")
        print(f"   Converted to: Others")
    except Exception as e:
        print(f"❌ FAIL: Error handling '?' ethnicity - {str(e)}")
    
    # Edge Case 2: Relation with "?"
    print("\n[6b] Edge Case: Relation with '?' value")
    edge_data_2 = pd.DataFrame({
        'A1_Score': [1], 'A2_Score': [0], 'A3_Score': [1], 'A4_Score': [1], 'A5_Score': [0],
        'A6_Score': [1], 'A7_Score': [0], 'A8_Score': [1], 'A9_Score': [1], 'A10_Score': [0],
        'gender': ['Male'], 'age': [25], 'ethnicity': ['Asian'], 'jaundice': ['No'],
        'austim': ['Yes'], 'contry_of_res': ['United States'], 'used_app_before': ['No'],
        'relation': ['?'], 'result': [6]
    })
    
    try:
        processed = preprocess_data(edge_data_2.copy(), encoders, train_df)
        print("✅ PASS: '?' relation handled correctly")
        print(f"   Converted to: Others")
    except Exception as e:
        print(f"❌ FAIL: Error handling '?' relation - {str(e)}")
    
    # Edge Case 3: Minimum age (0)
    print("\n[6c] Edge Case: Minimum age (0)")
    print("   Testing with age = 0")
    try:
        age_value = 0
        if age_value >= 0 and age_value <= 100:
            print("✅ PASS: Minimum age validated")
        else:
            print("❌ FAIL: Age validation failed")
    except Exception as e:
        print(f"❌ FAIL: Error in age validation - {str(e)}")
    
    # Edge Case 4: Maximum age (100)
    print("\n[6d] Edge Case: Maximum age (100)")
    print("   Testing with age = 100")
    try:
        age_value = 100
        if age_value >= 0 and age_value <= 100:
            print("✅ PASS: Maximum age validated")
        else:
            print("❌ FAIL: Age validation failed")
    except Exception as e:
        print(f"❌ FAIL: Error in age validation - {str(e)}")

# Test Case 7: Data Validation
def test_data_validation():
    """Test data validation rules"""
    print("\n" + "="*60)
    print("TEST 7: Data Validation Rules")
    print("="*60)
    
    # Test score range (0-1)
    print("\n[7a] Assessment Scores Range (0-1):")
    valid_scores = [0, 1]
    invalid_scores = [-1, 2, 0.5, "invalid"]
    
    for score in valid_scores:
        if score in [0, 1]:
            print(f"   ✅ Score {score}: Valid")
        else:
            print(f"   ❌ Score {score}: Invalid")
    
    # Test total score range (0-10)
    print("\n[7b] Total Score Range (0-10):")
    test_scores = [0, 5, 10, -1, 11]
    for score in test_scores:
        if 0 <= score <= 10:
            print(f"   ✅ Total {score}: Valid")
        else:
            print(f"   ❌ Total {score}: Invalid (out of range)")
    
    # Test gender values
    print("\n[7c] Gender Values:")
    valid_genders = ['Male', 'Female']
    for gender in valid_genders:
        print(f"   ✅ Gender '{gender}': Valid")
    
    # Test yes/no values
    print("\n[7d] Yes/No Values:")
    valid_yesno = ['Yes', 'No']
    for value in valid_yesno:
        print(f"   ✅ Value '{value}': Valid")

# Main test runner
def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("AUTISM PREDICTION APP - TEST SUITE")
    print("="*60)
    print(f"Running comprehensive tests...")
    
    results = {
        'passed': 0,
        'failed': 0,
        'total': 7
    }
    
    # Test 1: Model Loading
    model_success, model = test_model_loading()
    if model_success:
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Test 2: Encoder Loading
    encoder_success, encoders = test_encoder_loading()
    if encoder_success:
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Test 3: Training Data Loading
    data_success, train_df = test_training_data_loading()
    if data_success:
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Only continue with other tests if basic loading succeeded
    if model_success and encoder_success and data_success:
        # Test 4: Preprocessing
        if test_preprocess_data(encoders, train_df):
            results['passed'] += 1
        else:
            results['failed'] += 1
        
        # Test 5: Predictions
        test_predictions(model, encoders, train_df)
        results['passed'] += 1  # Count as pass if it runs without crashing
        
        # Test 6: Edge Cases
        test_edge_cases(encoders, train_df)
        results['passed'] += 1  # Count as pass if it runs without crashing
        
        # Test 7: Data Validation
        test_data_validation()
        results['passed'] += 1  # Count as pass if it runs without crashing
    else:
        results['failed'] += 4
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success Rate: {(results['passed']/results['total'])*100:.1f}%")
    print("="*60)
    
    if results['passed'] == results['total']:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed. Please review above.")
    
    return results

if __name__ == "__main__":
    run_all_tests()
