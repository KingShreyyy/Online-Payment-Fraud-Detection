import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("data/onlinefraud.csv.zip")

print("Dataset Head:")
print(data.head())
print("\nDataset Shape:")
print(data.shape)
print("\nMissing Values:")
print(data.isnull().sum())
print("\nTransaction Types:")
print(data["type"].value_counts())

figure = px.pie(data, names="type", title="Distribution of Transaction Types")
figure.show()

numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
if "isFraud" in correlation.columns:
    print("\nCorrelation with 'isFraud':")
    print(correlation["isFraud"].sort_values(ascending=False))
else:
    print("'isFraud' column not found in the dataset.")

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

data = data.drop(["nameOrig", "nameDest"], axis=1)

x = data.drop("isFraud", axis=1)
y = data["isFraud"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

print("\nModel Accuracy:", model.score(xtest, ytest))
print("\nClassification Report:")
print(classification_report(ytest, model.predict(xtest)))
print("\nConfusion Matrix:")
print(confusion_matrix(ytest, model.predict(xtest)))

features = np.array([[4, 1, 9000.60, 9000.60, 0.0, 0.0, 0.0, 0.0]])
features_df = pd.DataFrame(features, columns=x.columns)
print("\nPrediction for Features [4, 1, 9000.60, 9000.60, 0.0, 0.0, 0.0, 0.0]:", model.predict(features_df))
