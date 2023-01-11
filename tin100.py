
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
    


# Split datasett i train og test


train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
train = train_raw.drop(['Loan_ID'], axis = 1)
test = test_raw.drop(['Loan_ID'], axis = 1)


# Sjekker summen av NAN verdier i hver kollone for train
train.isnull().sum()

# Sjekker summen av NAN verdier i hver kollone for test
test.isnull().sum()


for col in train:
    imr = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imr = imr.fit(train[[f'{col}']])
    train[f'{col}'] = imr.transform(train[[f'{col}']])
    
for col in test:
    imr = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imr = imr.fit(test[[f'{col}']])
    test[f'{col}'] = imr.transform(test[[f'{col}']])


fig, ax  = plt.subplots(2,4,figsize=(16,10))
sns.countplot('Loan_Status', data = train, ax=ax[0][0] )
sns.countplot('Gender', data = train, ax=ax[0][1] )
sns.countplot('Married', data = train, ax=ax[0][2] )
sns.countplot('Education', data = train, ax=ax[0][3] )
sns.countplot('Self_Employed', data = train, ax=ax[1][0] )
sns.countplot('Dependents', data = train, ax=ax[1][1] )
sns.countplot('Property_Area', data = train, ax=ax[1][2] )
sns.countplot('Loan_Status', data = train, ax=ax[1][3] )


le = LabelEncoder()
for col in train[['Gender', 'Married', 'Education','Self_Employed', 'Dependents', 'Property_Area','Credit_History', 'Loan_Status']]:  
    #print(col)
    train[col] = le.fit_transform(train[col])
#
# Print df.head for checking the transformation
#

train['CoapplicantIncome'] = train['CoapplicantIncome'].astype('int')
train.head()


le = LabelEncoder()
for col in test[['Gender', 'Married', 'Education','Self_Employed', 'Dependents','Credit_History', 'Property_Area']]:  
    #print(col)
    test[col] = le.fit_transform(test[col])
#
# Print df.head for checking the transformation
#
test['CoapplicantIncome'] = test['CoapplicantIncome'].astype('int')
test.head()


fig, ax  = plt.subplots(2,4,figsize=(16,10))
sns.countplot('Loan_Status', data = train, ax=ax[0][0] )
sns.countplot('Gender', data = train, ax=ax[0][1] )
sns.countplot('Married', data = train, ax=ax[0][2] )
sns.countplot('Education', data = train, ax=ax[0][3] )
sns.countplot('Self_Employed', data = train, ax=ax[1][0] )
sns.countplot('Dependents', data = train, ax=ax[1][1] )
sns.countplot('Property_Area', data = train, ax=ax[1][2] )
sns.countplot('Loan_Status', data = train, ax=ax[1][3] )



#train.dtypes

x = train.drop('Loan_Status', axis = 1)
y = train['Loan_Status']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4, stratify=y) 


#x_train.shape

#x_test.shape


log_clf = LogisticRegression()

param_grid = [{'C': [0.01,0.1,1,10,100,1000],
              'penalty':['l2']}]
gs_r = GridSearchCV(log_clf,
                    param_grid,
                    scoring='f1',
                    cv=5,
                    verbose = 1,
                    n_jobs= -1)
gs_r.fit(x_train, y_train)
print(gs_r.best_score_)
print(gs_r.best_params_)


clf = gs_r.best_estimator_

gs_r.score(x_test, y_test)

model = clf.fit(x, y)
pred = model.predict(test)


test['Loan_Status'] = pred


New_loan_status = []
for i in range(len(test)):
    if test.iloc[i]['Loan_Status'] == 0:
        #print('hei')
        income = 5*12*(test.iloc[i]['ApplicantIncome'] + test.iloc[i]['CoapplicantIncome'])
        loan = (test.iloc[i]['LoanAmount'])*1000
        New_loan_status.append(income < loan)
    else:
        New_loan_status.append(1)


test['new_loan_status'] = New_loan_status

#test.head(25)

df = test['Loan_Status']
df.value_counts()

df = test['new_loan_status']
df.value_counts()

