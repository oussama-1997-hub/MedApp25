import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Streamlit app title
st.title("Prediction of Target Using Decision Tree 2222")
st.write("Please fill in the details below to predict the target.")

# Load the data
url = "https://raw.githubusercontent.com/oussama-1997-hub/MedApp25/main/BD%20sans%20encod%20stand.xlsx"
df = pd.read_excel(url, engine="openpyxl")

# Display the data in the app (optional)
st.write("Here is the dataset:")
st.dataframe(df.head())

# 1. Separate features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except the last one (target column)
y = df.iloc[:, -1]   # The last column (target)

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, criterion='gini', random_state=42)
clf.fit(X_train, y_train)

# 4. Predict the target using the trained model
y_pred = clf.predict(X_test)

# 5. Evaluate the model (accuracy, precision, recall, F1 score)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display model evaluation metrics
st.write(f"Model Accuracy: {accuracy:.2f}")
st.write(f"Model Precision: {precision:.2f}")
st.write(f"Model Recall: {recall:.2f}")
st.write(f"Model F1 Score: {f1:.2f}")

# Collect user input for features using Streamlit text_input
user_inputs = {}

for feature in X.columns:
    user_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert the user inputs to a DataFrame (needed for prediction)
user_inputs_df = pd.DataFrame(user_inputs, index=[0])

# 6. Make prediction based on user inputs
if st.button("Predict Target"):
    prediction = clf.predict(user_inputs_df)
    st.write(f"Predicted Target: {prediction[0]}")

# Optionally, you can display a classification report
if st.button("Show Classification Report"):
    report = classification_report(y_test, y_pred)
    st.text(report)
