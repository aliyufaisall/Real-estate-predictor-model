from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin


# Column indices for bedrooms and bathrooms
bed_ix, bath_ix = 2, 3 # Adjust these indices based on your DataFrame structure

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # First Principle: Ensure no division by zero
        bath_per_bed = X[:, bath_ix] / (X[:, bed_ix] + 1e-6) 
        return np.c_[X, bath_per_bed]

# Load the model and pipeline once when the server starts
# (Using absolute paths helps avoid errors)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'predictor', 'abuja_housing_model.pkl')
pipeline_path = os.path.join(BASE_DIR, 'predictor', 'preprocessing_pipeline.pkl')

model = joblib.load(model_path)
pipeline = joblib.load(pipeline_path)

def index(request):
    prediction = None
    
    if request.method == 'POST':
        # 1. Get data from the form
        data = {
            'bedrooms': int(request.POST.get('bedrooms')),
            'bathrooms': int(request.POST.get('bathrooms')),
            'parking_space': int(request.POST.get('parking_space')),
            'state': request.POST.get('state'),
            'town': request.POST.get('town'),
            'title': request.POST.get('title'),
        }
        
        # 2. Convert to DataFrame and Predict
        df = pd.DataFrame([data])
        prepared_data = pipeline.transform(df)
        log_price = model.predict(prepared_data)
        prediction = np.expm1(log_price)[0] # Get the first number

    return render(request, 'predictor/index.html', {'prediction': prediction})

