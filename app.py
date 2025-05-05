pip install -r requirements.txt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.title("Streamlit in Colab")
st.write("Hello from Google Colab!")

name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}!")


# 1. Charger les données
df = pd.read_excel("https://github.com/SondesHammami/MedApp25/blob/main/BD_ML_1target.xlsx", engine="openpyxl")
print(df)

# 2. Séparer X et y (la dernière colonne est la cible)
X = df.iloc[:, :-1]  # toutes les colonnes sauf la dernière
y = df.iloc[:, -1]   # dernière colonne

# 3. Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Créer et entraîner le modèle
clf = DecisionTreeClassifier(
    max_depth=None,               # profondeur maximale de l’arbre
    min_samples_split=2,         # nombre minimum d’échantillons pour un split
    criterion='gini',            # fonction d’impureté
    random_state=42
)
clf.fit(X_train, y_train)

# 5. Prédictions
y_pred = clf.predict(X_test)

# 6. Évaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' gère les classes déséquilibrées
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Affichage des résultats
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Rapport détaillé par classe
print("\nClassification Report:\n", classification_report(y_test, y_pred))
