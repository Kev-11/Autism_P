# <span style="font-family: Arial; font-size: 24px;">Austim_P</span>

<p style="font-family: Arial; font-size: 16px;">A machine learning model deployed using Streamlit to predict whether a person has autism based on given input features.</p>

## <span style="font-family: Arial; font-size: 20px;">Features</span>
- Uses a trained ML model (`best_model.pkl`) for prediction.
- Encodes categorical features using `encoders.pkl`.
- Includes a Jupyter Notebook (`model.ipynb`) for model training and evaluation.
- Web-based interface built with Streamlit (`streamlit_app.py`).
- Dataset files (`datasets/`) for training and testing.

## <span style="font-family: Arial; font-size: 20px;">Installation</span>

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kev-11/Austim_P.git
   cd Austim_P
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## <span style="font-family: Arial; font-size: 20px;">Usage</span>

### <span style="font-family: Arial; font-size: 18px;">Run the Streamlit App</span>
```bash
streamlit run streamlit_app.py
```
This will open the web interface where users can input data and get autism predictions.

### <span style="font-family: Arial; font-size: 18px;">Model Training</span>
To retrain the model or modify it, open `model.ipynb` in Jupyter Notebook and follow the steps inside.

## <span style="font-family: Arial; font-size: 20px;">Dataset</span>
The project includes a dataset in the `datasets/` directory:
- `train.csv` - Training data.
- `test.csv` - Testing data.
- `sample_submission.csv` - Example of expected output format.

## <span style="font-family: Arial; font-size: 20px;">Dependencies</span>
All required Python packages are listed in `requirements.txt`. The key dependencies include:
- `scikit-learn`
- `pandas`
- `numpy`
- `streamlit`

## <span style="font-family: Arial; font-size: 20px;">Contributing</span>
Feel free to fork this repository, create a branch, and submit a pull request for improvements.

## <span style="font-family: Arial; font-size: 20px;">License</span>
This project is under the MIT License.

