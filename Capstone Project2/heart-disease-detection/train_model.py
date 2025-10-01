# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/heart_disease_dataset.csv")  # make sure path is correct

# Show column names for debugging
print("✅ Columns in dataset:", df.columns.tolist())

# Automatically select last column as target
target_column = df.columns[-1]
print(f"✅ Using '{target_column}' as the target column.")

# Features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model to a pickle file
with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("🎉 Model trained successfully and saved as heart_disease_model.pkl")
