import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
#file_path = "data.xlsx"
df = pd.read_excel("data.xlsx")
print(df.isnull().sum())  # Check for missing values

# df = df.assign(
#     Department=df['Department'].fillna(df['Department'].mode()[0]),
#     priority2=df['priority2'].fillna(df['priority2'].mode()[0])
# )

#df.fillna("Unknown", inplace=True)  # Replace NaNs with "Unknown"
# Define Features (X) and Multi-Output Labels (Y)
X = df[['Query', 'Syllabus', 'Department', 'Database', 'College']]  # Input features
Y = df[['priority1', 'priority2']]  # Target columns

# Split into Training & Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Label Encoding (Handling Unseen Labels)
label_encoders = {}
for column in X_train.columns:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])

    # Handle unseen labels in X_test
    X_test[column] = X_test[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    
    label_encoders[column] = le  # Store encoders for future use

# Encode Output Columns (priority1, priority2)
for column in Y_train.columns:
    le = LabelEncoder()
    Y_train[column] = le.fit_transform(Y_train[column])
    Y_test[column] = Y_test[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)


# # Train a Multi-Output Random Forest Classifier
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, Y_train)


#XGBoost algorithm
# from xgboost import XGBClassifier

# # Train an XGBoost Classifier
# model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
# model.fit(X_train, Y_train)

# Make Predictions
Y_pred = model.predict(X_test)

# Evaluate Performance for Each Output
for i, col in enumerate(Y.columns):
    acc = accuracy_score(Y_test[col], Y_pred[:, i])
    print(f"Accuracy for {col}: {acc:.2f}")





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# # Load Dataset
# df = pd.read_excel("data.xlsx")
# print(df.isnull().sum())  # Check for missing values

# # Fill missing values with the most common category
# df['Department'].fillna(df['Department'].mode()[0], inplace=True)
# df['priority2'].fillna(df['priority2'].mode()[0], inplace=True)

# # Define Features (X) and Multi-Output Labels (Y)
# X = df[['Query', 'Syllabus', 'Department', 'Database', 'College']]
# Y = df[['priority1', 'priority2']]

# # Split into Training & Testing Sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # ✅ Label Encoding with Handling Unseen Labels
# label_encoders = {}
# for column in X_train.columns:
#     le = LabelEncoder()
#     X_train[column] = le.fit_transform(X_train[column].astype(str))
    
#     # Map unseen labels to -1
#     X_test[column] = X_test[column].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    
#     label_encoders[column] = le  # Store encoders for future use

# # ✅ Encode Output Columns (priority1, priority2)
# output_encoders = {}
# for column in Y_train.columns:
#     le = LabelEncoder()
#     Y_train[column] = le.fit_transform(Y_train[column].astype(str))
#     Y_test[column] = Y_test[column].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
#     output_encoders[column] = le  # Store for inverse transformation if needed

# # ✅ Train a Tuned Random Forest Classifier
# model = RandomForestClassifier(
#     n_estimators=500,    # Increase number of trees
#     max_depth=15,        # Increase max depth
#     min_samples_split=5, # Prevent overfitting
#     min_samples_leaf=2,  # More generalization
#     class_weight='balanced',  # Handle class imbalance
#     random_state=42
# )
# model.fit(X_train, Y_train)

# # Make Predictions
# Y_pred = model.predict(X_test)

# # ✅ Evaluate Performance for Each Output
# for i, col in enumerate(Y.columns):
#     acc = accuracy_score(Y_test[col], Y_pred[:, i])
#     print(f"Improved Accuracy for {col}: {acc:.2f}")
