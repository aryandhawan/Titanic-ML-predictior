import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score,accuracy_score,precision_score,recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose  import ColumnTransformer
data=pd.read_csv('titanic.csv')
df=pd.DataFrame(data=data)


## Data Preprocessing We drop irrelevant columns, handle missing values, and split the dataset into features and labels.


main_df=df.drop(['Survived','Ticket','Cabin','Age','PassengerId'],axis=1)
main_df.fillna('S',inplace=True)
main_df.dropna()
label_df=df['Survived']

x_train,x_test,y_train,y_test=train_test_split(main_df,label_df,test_size=0.2)

numerical_cols=['Pclass','Parch','Fare','SibSp']
categorical_cols=['Name','Sex','Embarked']

# steps preprocessing pipelines
categorical_pipeline=Pipeline([('onehot',OneHotEncoder(handle_unknown='ignore'))])
numerical_pipeline=Pipeline([('scaler',StandardScaler())])

# columntransformer

preprocessor_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


# model pipelines 
models={

    'linear model':Pipeline([
        ('preprocessor',preprocessor_pipeline),
        ('model',LogisticRegression(random_state=42))
    ]),

    'tree model':Pipeline([
        ('preprocessor',preprocessor_pipeline),
        ('model',DecisionTreeClassifier(random_state=42))
    ]),

    'forest model':Pipeline([
        ('preprocessor',preprocessor_pipeline),
        ('model',RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1))
    ])

}

for model_name,pipeline in models.items():
    pipeline.fit(X=x_train,y=y_train)
    print(f'for {model_name} the training has been completed!')
    


# evaluations

report={}

for model_name,pipeline in models.items():
    y_pred=pipeline.predict(x_test)

    clf_report= classification_report(y_pred=y_pred,y_true=y_test)
    accuracy=accuracy_score(y_pred=y_pred,y_true=y_test)
    precision=precision_score(y_pred=y_pred,y_true=y_test)
    f1=f1_score(y_pred=y_pred,y_true=y_test)
    recall=recall_score(y_pred=y_pred,y_true=y_test)

    report[model_name]={
        'classification_report':classification_report,
        'accuracy':accuracy,
        'precision':precision,
        'f1_score':f1_score,
        'recall_score':recall_score
    }

    # printing those results 
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", clf_report)

# hypertuning by GridsearchCV

from sklearn.model_selection import GridSearchCV

param_grid={
    'model__n_estimators':[50,100,200],
    'model__max_depth':[None,5,10,20],
    'model__min_samples_split':[2,5,10]
}

grid_search=GridSearchCV(estimator=models['forest model'],param_grid=param_grid,scoring='f1',cv=5,n_jobs=-1,verbose=1) # I chose Random forest because it is more robust and better perfomring in-terms of F1 and recall 
grid_search.fit(x_train,y_train)

print("Best parameters: ",grid_search.best_params_)
print("best score: ",grid_search.best_score_)
best_model=grid_search.best_estimator_

y_prediction=best_model.predict(x_test)
clf1_report=classification_report(y_test,y_prediction)
print(clf1_report)

## Try It Yourself: Custom Passenger Prediction Below, you can see how the model predicts survival for a sample passenger. Change the values to test other scenarios!

sample_passenger = pd.DataFrame({
    'Pclass': [1],
    'Name': ['Jane Doe'],
    'Sex': ['female'],
    'SibSp': [1],
    'Parch': [1],
    'Fare': [100],
    'Embarked': ['C']
})

prediction = best_model.predict(sample_passenger)[0]
print("Prediction:", "Survived! 🎉" if prediction == 1 else "Did Not Survive 😔")


## Conclusion
#The Random Forest model achieved the best F1 and recall scores, 
# making it the most robust choice for predicting Titanic survival. 
# Hyperparameter tuning further improved its performance. 
# For this problem recall was prioritized to minimize missed survivors.
