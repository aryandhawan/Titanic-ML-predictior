# Titanic-ML-predictior
Overview
This project tackles the classic Kaggle Titanic competition, where the goal is to predict whether a passenger survived the Titanic disaster based on features such as age, sex, ticket class, and more. The project demonstrates the full machine learning workflow: data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

# Dataset
The dataset contains information about Titanic passengers, including:

Pclass: Ticket class (1st, 2nd, 3rd)

Sex: Gender

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Fare: Ticket fare

Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
# Modelling Approach
Preprocessing:

Dropped irrelevant columns (Ticket, Cabin, PassengerId)

Handled missing values and encoded categorical variables

Scaled numerical features

# Models Used:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

# Evaluation:

Accuracy, Precision, Recall, F1 Score

Emphasis on recall (to minimise missing survivors)

Hyperparameter tuning with GridSearchCV

# Best Model:
Random Forest with tuned parameters, chosen for its robustness and superior F1/recall performance.

Results
Best Model: Random Forest Classifier

# Key Metrics:

Accuracy: ~0.80

Recall: Prioritised for survivor identification

F1 Score: Balanced performance

Please review the notebook for detailed classification reports and model comparisons.

I am open to contributions and suggestions for this project 

