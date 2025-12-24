import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv("onlinefraud.csv")

print("Dataset Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())
print("\nTransaction Types:\n", data["type"].value_counts())

fig = px.pie(data, names="type", title="Transaction Type Distribution")
fig.show()

data["type"] = data["type"].map({
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
})

data["isFraud"] = data["isFraud"].map({0: 0, 1: 1})

data.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1, inplace=True)

data["balance_diff_orig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
data["balance_diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]
data["amount_to_balance_ratio"] = data["amount"] / (data["oldbalanceOrg"] + 1)

data = data.sample(n=300000, random_state=42)

X = data.drop("isFraud", axis=1)
y = data["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"\n================ {model_name} ================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc
    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n========== MODEL COMPARISON ==========")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

sample_df = pd.DataFrame(
    np.zeros((1, X.shape[1])),
    columns=X.columns
)

sample_df = sample_df.astype(X.dtypes)

sample_df.loc[0, "type"] = 4
sample_df.loc[0, "amount"] = 9000.60
sample_df.loc[0, "oldbalanceOrg"] = 9000.60
sample_df.loc[0, "newbalanceOrig"] = 0.0
sample_df.loc[0, "oldbalanceDest"] = 0.0
sample_df.loc[0, "newbalanceDest"] = 0.0

best_model = models["Random Forest"]
prediction = best_model.predict(sample_df)

print("Sample Transaction Prediction:", "Fraud" if prediction[0] == 1 else "Not Fraud")
