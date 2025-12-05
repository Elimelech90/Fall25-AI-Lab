# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and preprocessing data
model = joblib.load('model_svc.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# TARGET = 'suicidal' — confirmed from your dataset
TARGET = 'suicidal'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input
        form_data = request.form.to_dict()
        
        # Create DataFrame
        input_df = pd.DataFrame([form_data])
        
        # Ensure correct feature order (exclude target 'suicidal')
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Encode each categorical feature using saved LabelEncoders
        for col in feature_columns:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen categories by using the most frequent class (index 0)
                input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                input_df[col] = le.transform(input_df[col])
        
        # Make prediction
        pred = model.predict(input_df)[0]
        
        # Decode prediction: 0 → "No", 1 → "Yes"
        result_label = label_encoders[TARGET].inverse_transform([pred])[0]
        
        # Generate user-friendly message
        if result_label == 'Yes':
            risk_message = "⚠️ High Risk: Immediate support is recommended."
        else:
            risk_message = "✅ Low Risk: No immediate concern detected."

        return render_template('index.html', prediction=risk_message)
    
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)