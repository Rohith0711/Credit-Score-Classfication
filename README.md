# 📊 Credit-Score-Classfication using Machine Learning

# 1. 🚀 Project Overview
This project explores the use of machine learning models to classify individuals’ credit scores based on financial and behavioral data. Traditional models like FICO are limited in scope and accuracy, prompting the need for more scalable and inclusive alternatives.

# 2.🎯 Objective
To build and compare multiple ML models that predict creditworthiness using a dataset of 100,000 records from this [dataset] (https://www.kaggle.com/datasets/ayushsharma0812/dataset-for-credit-score-classification/data)

# 3.🔍 Description
This project focuses on building and comparing machine learning models to classify individuals' credit scores into three categories: Good, Standard, and Poor. The goal is to develop a scalable and accurate credit scoring model that goes beyond traditional systems like FICO by using behavioral and financial features from real-world data.

We used the above dataset with 100,000 records and applied several classification models. The project involved data cleaning, exploratory data analysis, feature engineering, model training, and evaluation using performance metrics like accuracy, precision, and ROC-AUC.

# 4.📁 Repository Structure
```
credit-score-classification/
│
├── data/
│   └── credit_score.csv                 # Cleaned dataset (or .gitignore if large)
│
├── notebooks/
│   ├── data_cleaning.ipynb             # Data cleaning and preprocessing
│   ├── eda.ipynb                        # Exploratory data analysis
│   ├── knn_model.ipynb                 # K-Nearest Neighbors
│   ├── decision_tree.ipynb             # Decision Tree model
│   ├── random_forest.ipynb             # Random Forest model
│   ├── svm_model.ipynb                 # Support Vector Machine
│   ├── logistic_regression.ipynb       # Logistic Regression model
│   └── ensemble_model.ipynb            # Ensemble of SVM + LogReg
│
├── reports/
│   └── Final Project Report.pdf        # Detailed project report
│
├── images/
│   └── visuals and plots used in EDA and models
│
├── README.md                           # Project overview and setup instructions
├── requirements.txt                    # Python packages used
└── .gitignore                          # Files to ignore (e.g., large datasets)
```

# 5.🔧 Tools & Libraries
a) Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

b) Jupyter Notebook

c) Git/GitHub

d) Kaggle (for data)

# 6.🧠 Comparing Models Trained & its Performance
| Model                        | Accuracy    | Notes                                        |
| ---------------------------- | ----------- | -------------------------------------------- |
| Logistic Regression          | \~84.8%     | Simple baseline model                        |
| K-Nearest Neighbors          | \~82%       | Sensitive to scale and distance              |
| Decision Tree                | \~88.5%     | Prone to overfitting                         |
| Support Vector Machine (SVM) | \~86%       | Performs well with clear margin              |
| **Random Forest**            | **\~89.4%** | Best performer, high accuracy and robustness |
| Ensemble (SVM + LogReg)      | \~87.5%     | Combines generalization and precision        |

# 📈 7. Key Highlights
Data cleaning with anomaly handling and IQR filtering

Dimensionality reduction through engineered one-hot encoding

Comparative model performance analysis using ROC-AUC and accuracy

Random Forest achieved the highest accuracy and robustness

# 8. EDA
<img width="368" alt="image" src="https://github.com/user-attachments/assets/0e3e37b5-2934-4e32-8d16-130a06d8c692" />    
<br>
<img height="150" width="400" alt="image" src="https://github.com/user-attachments/assets/842768e4-1bd4-4840-8cf5-fd31d1fbfbc9" />



# 9.🛠️ How to Run the Project
i) Clone the repository:
```
git clone https://github.com/<your-username>/credit-score-classification.git
cd credit-score-classification
```
ii) Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
iii) Install the required packages:
```pip install -r requirements.txt```

iv) Open notebooks:
```jupyter notebook```
