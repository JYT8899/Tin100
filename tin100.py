import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from numpy import interp

### Split datasett i train og test

train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
train = train_raw.drop(['Loan_ID'], axis=1)
test = test_raw.drop(['Loan_ID'], axis=1)

# Sjekker summen av NAN verdier i hver kollone for train
# train.isnull().sum()

# Sjekker summen av NAN verdier i hver kollone for test
# test.isnull().sum()

for col in train:
    imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imr = imr.fit(train[[f'{col}']])
    train[f'{col}'] = imr.transform(train[[f'{col}']])

for col in test:
    imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imr = imr.fit(test[[f'{col}']])
    test[f'{col}'] = imr.transform(test[[f'{col}']])

# fig, ax  = plt.subplots(2,4,figsize=(16,10))
# sns.countplot('Loan_Status', data = train, ax=ax[0][0] )
# sns.countplot('Gender', data = train, ax=ax[0][1] )
# sns.countplot('Married', data = train, ax=ax[0][2] )
# sns.countplot('Education', data = train, ax=ax[0][3] )
# sns.countplot('Self_Employed', data = train, ax=ax[1][0] )
# sns.countplot('Dependents', data = train, ax=ax[1][1] )
# sns.countplot('Property_Area', data = train, ax=ax[1][2] )
# sns.countplot('Loan_Status', data = train, ax=ax[1][3] )

le = LabelEncoder()
for col in train[
    ['Gender', 'Married', 'Education', 'Self_Employed', 'Dependents', 'Property_Area', 'Credit_History',
     'Loan_Status']]:
    # print(col)
    train[col] = le.fit_transform(train[col])
#
# Print df.head for checking the transformation
#

train['CoapplicantIncome'] = train['CoapplicantIncome'].astype('int')
# train.head()

le = LabelEncoder()
for col in test[
    ['Gender', 'Married', 'Education', 'Self_Employed', 'Dependents', 'Credit_History', 'Property_Area']]:
    # print(col)
    test[col] = le.fit_transform(test[col])
#
# Print df.head for checking the transformation
#
test['CoapplicantIncome'] = test['CoapplicantIncome'].astype('int')
# test.head()

# fig, ax  = plt.subplots(2,4,figsize=(16,10))
# sns.countplot('Loan_Status', data = train, ax=ax[0][0] )
# sns.countplot('Gender', data = train, ax=ax[0][1] )
# sns.countplot('Married', data = train, ax=ax[0][2] )
# sns.countplot('Education', data = train, ax=ax[0][3] )
# sns.countplot('Self_Employed', data = train, ax=ax[1][0] )
# sns.countplot('Dependents', data = train, ax=ax[1][1] )
# sns.countplot('Property_Area', data = train, ax=ax[1][2] )
# sns.countplot('Loan_Status', data = train, ax=ax[1][3] )

# train.dtypes

x = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y)


### Randomforest classifier

def RanForClf():
    parameter_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    rfc = RandomForestClassifier(random_state=42)
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=parameter_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    CV_rfc.fit(x, y)
    return CV_rfc
    #print('Potensielt best score ut ifra train datasettet: ', CV_rfc.best_score_)
    # CV_rfc.best_params_

    #y_pred_data = CV_rfc.predict(data)

    #return "Søknaden er godkjent hvis output er (1) og ikke godkjent for (0): ", y_pred_data



def predic(x, model):
    return model.predict(x)

### Gridsearch for feature importance
#
# param_ran_grid = {
#     'n_estimators': [100, 200, 300, 400, 500, 600],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [4, 5, 6, 7, 8],
#     'criterion': ['gini', 'entropy']
# }
# rfc = RandomForestClassifier(random_state=42)
# CV_rfc_ran = GridSearchCV(estimator=rfc, param_grid=param_ran_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
# CV_rfc_ran.fit(x_train, y_train)

## Feature importance
#
# feat_labels = np.array(train.columns)
#
# ran = RandomForestClassifier(random_state=42,
#                              max_features=CV_rfc_ran.best_params_['max_features'],
#                              n_estimators=CV_rfc_ran.best_params_['n_estimators'],
#                              max_depth=CV_rfc_ran.best_params_['max_depth'],
#                              criterion=CV_rfc_ran.best_params_['criterion'])
# ran.fit(x, y)
#
# importances = ran.feature_importances_
#
# indices = np.argsort(importances)[::-1]
#
# # Print results on screen
# for f in range(x_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30,
#                             feat_labels[indices[f]],
#                             importances[indices[f]]))
#
# plt.title('Feature Importance')
# plt.bar(range(x_train.shape[1]),
#         importances[indices],
#         align='center')
#
# plt.xticks(range(x_train.shape[1]),
#            feat_labels[indices], rotation=90)
# plt.xlim([-1, x_train.shape[1]])
# plt.tight_layout()
# plt.show()

### Predict of the whole datasett
if __name__ == "__main__":

    ### Preprocesing

    ## korrelation matrise
    plt.figure(figsize=(12, 6))

    sns.heatmap(train.corr(), cmap='BrBG', fmt='.2f',
                linewidths=2, annot=True)

    ## Bar plot 1
    sns.catplot(x="Gender", y="Married",
                hue="Loan_Status",
                kind="bar",
                data=train)

    ## Bar plot 2
    sns.catplot(y="LoanAmount", x="Gender",
                hue="Loan_Status",
                kind="bar",
                data=train)

    ### Logistic regression

    # log_clf = LogisticRegression()
    #
    # param_grid = [{'C': [0.01, 0.1, 1, 10, 100, 1000],
    #                'penalty': ['l2']}]
    # gs_r = GridSearchCV(log_clf,
    #                     param_grid,
    #                     scoring='f1',
    #                     cv=5,
    #                     verbose=1,
    #                     n_jobs=-1)
    # gs_r.fit(x_train, y_train)
    # print(gs_r.best_score_)
    # print(gs_r.best_params_)
    #
    # clf = gs_r.best_estimator_
    #
    # gs_r.score(x_test, y_test)

    model = RanForClf()
    pred = model.predict(test)

    test['Loan_Status'] = pred

    ### Creating a new column with 5* total income < loan

    New_loan_status = []
    for i in range(len(test)):
        if test.iloc[i]['Loan_Status'] == 0:
            # print('hei')
            income = 5 * 12 * (test.iloc[i]['ApplicantIncome'] + test.iloc[i]['CoapplicantIncome'])
            loan = (test.iloc[i]['LoanAmount']) * 1000
            New_loan_status.append(income < loan)
        else:
            New_loan_status.append(1)

    test['new_loan_status'] = New_loan_status

    # test.head(25)

    df = test['Loan_Status']
    df.value_counts()

    df = test['new_loan_status']
    df.value_counts()
