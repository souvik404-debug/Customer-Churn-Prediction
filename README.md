# Customer Churn Prediction Using Machine Learning

A Machine Learning project that predicts whether a telecom customer is likely to leave a company using customer demographics, subscription details, services, and billing information.

## Project Overview

Customer churn is a major challenge for telecom companies. This project uses machine learning algorithms to identify customers who are at risk of leaving and provides customer retention recommendations.

## Objectives

- Predict customer churn using machine learning.
- Compare different classification algorithms.
- Improve churn detection using hyperparameter tuning.
- Generate churn probability for individual customers.
- Provide actionable customer retention recommendations.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

**Telco Customer Churn Dataset**

File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information, including demographics, tenure, internet services, contract details, monthly charges, total charges, and churn status.

## Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Tuned Logistic Regression

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.48% | 0.65 | 0.56 | 0.60 |
| Decision Tree | 72.60% | 0.48 | 0.48 | 0.48 |
| Random Forest | 78.50% | 0.62 | 0.50 | 0.55 |
| Tuned Logistic Regression | 74.02% | 0.51 | 0.79 | 0.62 |

### Best Model

Tuned Logistic Regression achieved the highest F1 Score of **0.62** and a churn recall of **0.79**.

Logistic Regression achieved the highest overall accuracy of **80.48%**.

## Key Features

- Data preprocessing and cleaning.
- Exploratory data analysis.
- Feature engineering.
- Model training and evaluation.
- Hyperparameter tuning.
- Confusion matrix visualization.
- Customer churn probability prediction.
- Customer retention recommendations.

## How to Run the Project

**Step 1: Clone the repository**

```bash
git clone https://github.com/souvik404-debug/Customer-Churn-Prediction.git
```

**Step 2: Open the project folder**

```bash
cd Customer-Churn-Prediction
```

**Step 3: Install the required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Run the Python program**

```bash
python main.py
```

Make sure the dataset CSV file is present in the project directory.

## Sample Prediction Output

The system predicts whether a customer is likely to churn and displays:

- Churn prediction.
- Customer risk status.
- Churn probability.
- Personalized customer retention recommendations.

## Future Improvements

- Develop a web-based user interface.
- Integrate real-time customer data.
- Improve prediction performance.
- Deploy the model as a web application.

## Author

**Souvik Giri**

GitHub: https://github.com/souvik404-debug

LinkedIn: https://www.linkedin.com/in/souvik-giri-b3242a391

---

⭐ If you find this project useful, consider giving the repository a star!
