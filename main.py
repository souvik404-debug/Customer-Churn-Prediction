import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# ==================================
# PHASE 2: EXPLORATORY DATA ANALYSIS
# ==================================

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check unique values in each column
print("\nUnique Values:")
print(df.nunique())

# Check customer churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Check churn percentage
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
# PHASE 2: EXPLORATORY DATA ANALYSIS

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique Values:")
print(df.nunique())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
# ==================================
# PHASE 3: DATA CLEANING
# ==================================

# Check duplicate customer IDs
print("\nDuplicate Customer IDs:")
print(df["customerID"].duplicated().sum())

# Check blank values
print("\nBlank Values:")
print((df == " ").sum())

# Check TotalCharges data type
print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values again
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# Check dataset shape
print("\nDataset Shape After Cleaning Check:")
print(df.shape)
# Handle missing TotalCharges values

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Verify missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check final dataset shape
print("\nFinal Dataset Shape:")
print(df.shape)

# Check TotalCharges data type
print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)
# ==================================
# PHASE 4: DATA VISUALIZATION
# ==================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# 1. Customer Churn Distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()


# 2. Churn by Gender
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="gender", hue="Churn")

plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.show()


# 3. Churn by Contract Type
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.show()


# 4. Churn by Internet Service
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="InternetService", hue="Churn")

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.show()


# 5. Monthly Charges vs Churn
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.show()

# ==========================================
# PHASE 5: FEATURE ENGINEERING
# ==========================================

# Remove unnecessary customer ID column
df = df.drop("customerID", axis=1)

# Convert categorical columns into numerical columns
df = pd.get_dummies(df, drop_first=True)

# Display the processed dataset
print("\nProcessed Dataset:")
print(df.head())

# Display new dataset shape
print("\nProcessed Dataset Shape:")
print(df.shape)

# Separate features and target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# PHASE 6: TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display dataset shapes
print("\nTraining and Testing Data:")

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# ==========================================
# PHASE 7: MODEL TRAINING
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train all models
for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)
    print(name, "Training Completed!")

# ==========================================
# PHASE 8: MODEL EVALUATION
# ==========================================

from sklearn.metrics import accuracy_score, classification_report

print("\nMODEL EVALUATION RESULTS")

for name, model in models.items():

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n------------------------------")
    print("Model:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# ==========================================
# PHASE 9: MODEL COMPARISON
# ==========================================

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, pos_label=True), 3),
        "Recall": round(recall_score(y_test, y_pred, pos_label=True), 3),
        "F1 Score": round(f1_score(y_test, y_pred, pos_label=True), 3)
    })

# Create comparison table
comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON:")
print(comparison.to_string(index=False))

# Find best model based on F1 Score
best_model = comparison.loc[comparison["F1 Score"].idxmax()]

print("\nBest Model:", best_model["Model"])
print("Best F1 Score:", best_model["F1 Score"])

# ==========================================
# PHASE 10: CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

for name, model in models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[False, True])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    disp.plot(cmap="Blues", values_format="d")

    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ==========================================
# PHASE 11: HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=2000))
])

# Set parameters
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__class_weight': [None, 'balanced']
}

# Apply Grid Search
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Results
print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Recall:")
print(round(grid.best_score_ * 100, 2), "%")

print("\nTuned Model Accuracy:")
print(round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# PHASE 12: TUNED MODEL CONFUSION MATRIX
# ==========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[False, True])

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Tuned Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()


# ==========================================
# PHASE 13: FINAL MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Combine original models with tuned model
all_models = {
    **models,
    "Tuned Logistic Regression": best_model
}

final_results = []

for name, clf in all_models.items():

    predictions = clf.predict(X_test)

    final_results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(
            y_test, predictions) * 100, 2),

        "Precision": round(precision_score(
            y_test, predictions, zero_division=0), 2),

        "Recall": round(recall_score(
            y_test, predictions, zero_division=0), 2),

        "F1 Score": round(f1_score(
            y_test, predictions, zero_division=0), 2)
    })

# Create comparison table
final_comparison = pd.DataFrame(final_results)

print("\nFINAL MODEL PERFORMANCE COMPARISON")
print(final_comparison.to_string(index=False))

# Find best model based on F1 Score
best_index = final_comparison["F1 Score"].idxmax()

print("\nBest Model:",
      final_comparison.loc[best_index, "Model"])

print("Best F1 Score:",
      final_comparison.loc[best_index, "F1 Score"])


# ==========================================
# PHASE 14: CUSTOMER CHURN PREDICTION SYSTEM
# ==========================================

print("\n========== CUSTOMER CHURN PREDICTION ==========")

# Take customer details
gender = input("Enter Gender (Male/Female): ")
SeniorCitizen = int(input("Senior Citizen (0/1): "))
Partner = input("Partner (Yes/No): ")
Dependents = input("Dependents (Yes/No): ")
tenure = int(input("Tenure (months): "))
PhoneService = input("Phone Service (Yes/No): ")
MultipleLines = input("Multiple Lines (Yes/No/No phone service): ")
InternetService = input("Internet Service (DSL/Fiber optic/No): ")
OnlineSecurity = input("Online Security (Yes/No/No internet service): ")
OnlineBackup = input("Online Backup (Yes/No/No internet service): ")
DeviceProtection = input("Device Protection (Yes/No/No internet service): ")
TechSupport = input("Tech Support (Yes/No/No internet service): ")
StreamingTV = input("Streaming TV (Yes/No/No internet service): ")
StreamingMovies = input("Streaming Movies (Yes/No/No internet service): ")
Contract = input("Contract (Month-to-month/One year/Two year): ")
PaperlessBilling = input("Paperless Billing (Yes/No): ")
PaymentMethod = input("Payment Method: ")
MonthlyCharges = float(input("Monthly Charges: "))
TotalCharges = float(input("Total Charges: "))

# Create customer DataFrame
new_customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

# Convert categorical data into numerical data
new_customer_encoded = pd.get_dummies(new_customer)

# Match training dataset columns
new_customer_encoded = new_customer_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

# Predict customer churn
prediction = best_model.predict(new_customer_encoded)

# Display result
print("\n========== PREDICTION RESULT ==========")

if prediction[0] == True:
    print("Prediction: CUSTOMER WILL CHURN")
    print("Status: Customer is at risk of leaving.")
else:
    print("Prediction: CUSTOMER WILL NOT CHURN")
    print("Status: Customer is likely to stay.")

# Display prediction probability
probability = best_model.predict_proba(new_customer_encoded)[0]

churn_index = list(best_model.classes_).index(True)

print("Churn Probability:", round(probability[churn_index] * 100, 2), "%")

print("\n========== END OF PREDICTION ==========")


# ==========================================
# PHASE 15: CUSTOMER CHURN PREVENTION SYSTEM
# ==========================================

print("\n========== CUSTOMER RETENTION RECOMMENDATIONS ==========")

if prediction[0] == True:

    print("\nCustomer is at risk of leaving.")
    print("\nRecommended Actions:")

    if Contract == "Month-to-month":
        print("1. Offer a long-term contract discount.")

    if tenure < 12:
        print("2. Provide a special welcome and loyalty offer.")

    if TechSupport == "No":
        print("3. Offer free technical support.")

    if MonthlyCharges > 80:
        print("4. Provide a personalized discount on monthly charges.")

    if InternetService == "Fiber optic":
        print("5. Check internet service quality and customer satisfaction.")

    print("\nGoal: Improve customer satisfaction and prevent churn.")

else:

    print("\nCustomer is not currently predicted to churn.")
    print("Recommended Action: Maintain good customer service.")
    print("Offer loyalty rewards to retain the customer.")

print("\n========== END OF RECOMMENDATIONS ==========")
