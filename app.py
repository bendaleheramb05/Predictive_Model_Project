# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# -------------------------
# Load Dataset
# -------------------------
print("Program Started...")
df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset Loaded Successfully")
print(df.head())
print("First 5 Rows")
print(df.head())

# -------------------------
# Data Cleaning
# -------------------------

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop('Cabin', axis=1, inplace=True)

# -------------------------
# Encode Categorical Data
# -------------------------

encoder = LabelEncoder()

df['Sex'] = encoder.fit_transform(df['Sex'])

df['Embarked'] = encoder.fit_transform(df['Embarked'])

# -------------------------
# Select Features
# -------------------------

X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

y = df['Survived']

# -------------------------
# Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------
# Decision Tree Model
# -------------------------

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy")

print(accuracy_score(y_test, dt_pred))

# -------------------------
# Random Forest Model
# -------------------------

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy")

print(accuracy_score(y_test, rf_pred))

# -------------------------
# Classification Report
# -------------------------

print("\nClassification Report")

print(classification_report(y_test, rf_pred))

# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# -------------------------
# ROC Curve
# -------------------------

y_prob = rf.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="AUC = %.2f"%roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()