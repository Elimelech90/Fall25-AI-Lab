# train_model.py
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Define column names (based on your dataset structure)
columns = [
    'id', 'gender', 'age', 'city', 'occupation', 'academic_stress',
    'financial_stress', 'sleep_hours_numeric', 'study_load', 'depression',
    'sleep_duration', 'diet', 'degree', 'therapy', 'social_support',
    'extracurricular', 'suicidal'
]

# Load the dataset
df = pd.read_csv('student_depression_dataset.csv', names=columns, header=None)

# Drop 'id' (not useful for prediction)
df = df.drop(columns=['id'])

#  CHOOSE YOUR TARGET VARIABLE HERE
# Options: 'gender', 'depression', or 'suicidal'
target = 'suicidal'  # ← CHANGE THIS IF NEEDED
y = df[target]
X = df.drop(columns=[target])

# Encode all categorical columns (including target)
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns.tolist() + [target]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Update X and y after encoding
y = df[target]
X = df.drop(columns=[target])

# Train SVM model
model = SVC(probability=True)
model.fit(X, y)

# Save all files
joblib.dump(model, 'model_svc.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print(" Training complete! Saved:")
print(" - model_svc.pkl")
print(" - label_encoders.pkl")
print(" - feature_columns.pkl")