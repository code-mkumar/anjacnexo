import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import joblib  # For saving the model
import os
# Load Dataset
file_path = "data.xlsx"
df = pd.read_excel(file_path)
df.fillna("Unknown", inplace=True)

# Define Features and Target Columns
categorical_cols = ["Query", "College", "Department", "Database", "Syllabus"]
target_cols = ["priority1", "priority2"]

# Initialize Encoders
onehotencoder = OneHotEncoder(handle_unknown="ignore")
labelencoders = {col: LabelEncoder() for col in target_cols}

# Transform categorical input features (X)
X = onehotencoder.fit_transform(df[categorical_cols])

# Encode target labels (y)
y = df[target_cols].apply(lambda col: labelencoders[col.name].fit_transform(col))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Multi-Output Random Forest Model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_model, n_jobs=-1)
model.fit(X_train, y_train)

# Save Model & Encoders
train_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/trained_model.pkl"))
ohe = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/onehot_encoder.pkl"))
label_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/label_encoders.pkl"))
joblib.dump(model, train_file)
joblib.dump(onehotencoder, ohe)
joblib.dump(labelencoders, label_file)

# ✅ Prediction Function
def predict_priority(data_dict):
    """Predict the priority columns based on input data."""
    # Load Model & Encoders
    model = joblib.load(train_file)
    onehotencoder = joblib.load(ohe)
    labelencoders = joblib.load(label_file)

    # Convert input data to DataFrame
    df_new = pd.DataFrame([data_dict])

    # Ensure missing columns are added
    for col in categorical_cols:
        if col not in df_new.columns:
            df_new[col] = "Unknown"

    # Transform input data
    print(categorical_cols)
    print(df_new)
    # print(df_new[categorical_cols].applymap(type))  # Show data types of each cell
    # print(df_new[categorical_cols].head())  # Show first few rows

    df_new[categorical_cols] = df_new[categorical_cols].applymap(
        lambda x: ','.join(x) if isinstance(x, list) else x
    )
    X_new = onehotencoder.transform(df_new[categorical_cols])

    # Predict Priority Labels
    prediction = model.predict(X_new)

    # Convert Predictions Back to Original Labels
    priority1_pred = labelencoders["priority1"].inverse_transform([prediction[0][0]])[0]
    priority2_pred = labelencoders["priority2"].inverse_transform([prediction[0][1]])[0]

    return priority1_pred, priority2_pred  # Return actual category names