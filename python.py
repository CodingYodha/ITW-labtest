# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor for regression
from sklearn.metrics import accuracy_score, classification_report  # change for regression

# Load your dataset
df = pd.read_csv("your_dataset.csv")  # Replace with your file

# Separate features and target
X = df.drop("target_column", axis=1)  # Replace 'target_column' with your label column
y = df["target_column"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier()  # Change to RandomForestRegressor() for regression

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model (optional)
import joblib
joblib.dump(model, "trained_model.pkl")
