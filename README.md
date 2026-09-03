# 📞 Logistic Regression on Telco Customer Churn

A Machine Learning project that predicts **customer churn** using Logistic Regression and compares the performance of multiple classification algorithms.

## 📌 Project Overview

Customer churn occurs when a customer stops using a company's service.

This project uses the **Telco Customer Churn dataset** to analyze customer information and predict whether a customer is likely to churn.

### 🎯 Objectives

* Load and explore the Telco Customer Churn dataset
* Perform Exploratory Data Analysis (EDA)
* Handle missing values
* Convert categorical variables into numerical variables
* Split the dataset into training and testing data
* Build a Logistic Regression model
* Evaluate the model using classification metrics
* Generate a Confusion Matrix
* Generate an ROC curve
* Compare multiple machine learning models

---

## 🛠️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab / Jupyter Notebook

---

## 📊 Dataset

### Telco Customer Churn Dataset

The project reads the Telco Customer Churn CSV dataset:

```text
WA_Fn-UseC_-Telco-Customer-Churn (1).csv
```

The target variable is:

```text
Churn
```

The project converts the churn labels into numerical values using `LabelEncoder`.

---

# 🔍 Exploratory Data Analysis

The project performs several EDA operations.

### Data Exploration

* Dataset shape
* Dataset size
* Data types
* First and last records
* Descriptive statistics
* Churn distribution

### Numeric Variables

The following variables are analyzed:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Histograms are created to understand their distributions.

### Categorical Variables

The project analyzes variables such as:

* Gender
* Senior Citizen
* Partner
* Dependents
* Phone Service
* Multiple Lines
* Internet Service
* Contract
* Paperless Billing
* Payment Method

Count plots and stacked bar charts are generated to study their relationship with churn.

---

# 🧹 Data Preprocessing

### Handling Missing Values

`TotalCharges` is converted into a numeric variable and missing values are replaced with the mean.

```python
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].mean()
)
```

### Encoding Categorical Variables

Categorical variables are converted into numerical dummy variables using Pandas `get_dummies()`.

```python
x_num1 = pd.get_dummies(x_catg)
```

The numerical and categorical features are then combined into a single feature dataset.

---

# 🤖 Logistic Regression

The main model used in this project is **Logistic Regression**.

```python
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(max_iter=50000)

log.fit(train_x, train_y)
```

The dataset is divided into:

* **75% Training Data**
* **25% Testing Data**

using `train_test_split`.

---

# 📈 Model Evaluation

The Logistic Regression model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC
* Confusion Matrix
* ROC Curve

### Confusion Matrix

A heatmap is used to visualize the confusion matrix.

```text
                 Predicted
               0          1
Actual  0     TN         FP
        1     FN         TP
```

---

# 📉 ROC Curve

The project generates an ROC curve to visualize the model's classification performance.

The False Positive Rate and True Positive Rate are calculated using `roc_curve`, and the AUC is calculated using `auc`.

---

# 🏆 Model Comparison

The project also trains and compares several classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine
6. AdaBoost

The accuracy of each model is stored and displayed for comparison.

```python
acc_df = pd.DataFrame(
    list(accuracy_dict.items()),
    columns=['Model', 'Accuracy']
)
```

---

## 🔄 Machine Learning Workflow

```text
Telco Customer Dataset
          ↓
     Data Cleaning
          ↓
    Exploratory Analysis
          ↓
  Missing Value Handling
          ↓
Categorical Encoding
          ↓
    Train-Test Split
          ↓
 Logistic Regression
          ↓
    Model Prediction
          ↓
 Model Evaluation
          ↓
ROC / Confusion Matrix
          ↓
 Model Comparison
```

---

## 📂 Project Structure

```text
Logistic-Regression-Telco-Churn/
│
├── logistic_regression_on_telco_dataset.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Logistic-Regression-Telco-Churn.git
```

### 2. Open the project

```bash
cd Logistic-Regression-Telco-Churn
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the program

```bash
python logistic_regression_on_telco_dataset.py
```

---

## 📦 Requirements

Create a `requirements.txt` file:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🎓 Learning Outcomes

This project helps demonstrate:

* Supervised Machine Learning
* Logistic Regression
* Binary Classification
* Exploratory Data Analysis
* Data Cleaning
* Missing Value Imputation
* Label Encoding
* One-Hot Encoding
* Train-Test Split
* Classification Metrics
* Confusion Matrix
* ROC Curve
* AUC
* Model Comparison

---

## 👨‍💻 Author

**Mahadev Prasad L**

🎓 Artificial Intelligence & Data Science
🏫 Maharaja Institute of Technology Thandavapura
📚 3rd Year – 5th Semester
🎓 VTU Student

---

## ⭐ Project

This project is created for **academic and learning purposes** as part of Machine Learning studies.

If you find this project useful, please ⭐ **star the repository**.
