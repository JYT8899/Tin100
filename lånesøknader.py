#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:27:40 2022

@author: gurubaranrajeshwaran
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import LogisticRegression


X = []

def train_model():

    #Reading csv files
    Train_data = pd.read_csv("train.csv")
    Test_data = pd.read_csv("test.csv")

    Train_data = Train_data.drop(['Loan_ID'], axis=1)

    Test_data = Test_data.drop(['Loan_ID'], axis=1)
    #Cleaning the training data

    for col in Train_data:    
        imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imr = imr.fit(Train_data[[f"{col}"]])
        Train_data[f"{col}"] = imr.transform(Train_data[[f"{col}"]])

    for col in Test_data:    
        imr = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imr = imr.fit(Test_data[[f"{col}"]])
        Test_data[f"{col}"] = imr.transform(Test_data[[f"{col}"]])

    Train_data["Gender"]= Train_data["Gender"].map({"Male":0, "Female":1})
    Train_data["Married"]= Train_data["Married"].map({"No":0, "Yes":1})
    Train_data["Education"]= Train_data["Education"].map({"Not Graduate":0, "Graduate":1})
    Train_data["Self_Employed"]= Train_data["Self_Employed"].map({"No":0, "Yes":1})
    Train_data["Dependents"]= Train_data["Dependents"].map({"0":0, "1":1, "2":2, "3+":3})
    Train_data["Property_Area"]= Train_data["Property_Area"].map({"Urban":0, "Semiurban":1, "Rural":2})
    Train_data["Loan_Status"]= Train_data["Loan_Status"].map({"N":0, "Y":1})

    Test_data["Gender"]= Test_data["Gender"].map({"Male":0, "Female":1})
    Test_data["Married"]= Test_data["Married"].map({"No":0, "Yes":1})
    Test_data["Education"]= Test_data["Education"].map({"Not Graduate":0, "Graduate":1})
    Test_data["Self_Employed"]= Test_data["Self_Employed"].map({"No":0, "Yes":1})
    Test_data["Dependents"]= Test_data["Dependents"].map({"0":0, "1":1, "2":2, "3+":3})
    Test_data["Property_Area"]= Test_data["Property_Area"].map({"Urban":0, "Semiurban":1, "Rural":2})

    X = Train_data.drop("Loan_Status", axis=1).values
    y = Train_data["Loan_Status"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

    GBr = LogisticRegression()
    param_grid = [{'C': [0.001, 0.01 ,0.1, 1, 100, 1000],
                   'penalty' : ['l1', 'l2']}]
    gs_r = GridSearchCV(GBr,
                    param_grid, 
                    scoring='accuracy', 
                    cv=2,
                    verbose=1,
                   n_jobs=-1)
    gs_r.fit(X_train, y_train)
    clf = gs_r
    clf.fit(X, y)
    return clf 

model = train_model()

def predict(x, model):
    return model.predict(x)

    
st.write("# F?? tilbud p?? l??n fra oss")

st.header("Fyll ut s??knaden s?? gir vi deg et svar basert p?? din informasjon")
st.write("Verdier skal oppgis i USD")

Kj??nn = st.selectbox("Kj??nn", ("-velg-", "Mann", "Kvinne"))
Gift = st.selectbox("Gift", ("-velg-", "Ja", "Nei"))
Utdannelse = st.selectbox("Har du en utdannelse?", ("-velg-", "Ja", "Nei"))
Selvstendig_n??ringsdrivende = st.selectbox("Selvstendig n??ringsdrivende", ("-velg-", "Ja", "Nei"))
Barn_under_18 = st.selectbox("Hvor mange barn under 18 ??r?",("-velg-", "0", "1", "2", "3+"))
Eiendomsomr??det = st.selectbox("Eiendomsomr??det", ("-velg-", "Urban", "Semiurban", "Landlig"))
Kreditt = st.selectbox("Har du tidligere betalingshistorikk?", ("-velg-", "Ja", "Nei"))
Inntekt = st.number_input("Hva er inntekten din?")
Meds??kerinntekt = st.number_input("Hva er meds??kers inntekt, hvis ingen meds??ker skriv 0?")
L??n = st.number_input("Hvor mye vil du l??ne? (Verdier er i 1000)")
L??netid = st.number_input("Hvor lenge vil du l??ne? (oppgi antall mnd)")

Kj??nn_doct = {"Mann":0, "Kvinne":1}
Gift_doct = {"Ja":1, "Nei":0}
Utdannelse_doct = {"Ja":1, "Nei":0}
Selvstendig_n??ringsdrivende_doct = {"Ja":1, "Nei":0}
Barn_under_18_doct = {"0":0, "1":1, "2":2, "3+":3}
eiendomsomr??de_doct = {"Urban":0, "Semiurban":1, "Landlig":2}
Kreditt_doct = {"Ja":1.0, "Nei":0.0}
try:
    Kj??nn_svar = Kj??nn_doct[Kj??nn]
    Gift_svar = Gift_doct[Gift]
    Utdannelse_svar = Utdannelse_doct[Utdannelse]
    Selvstendig_n??ringsdrivende_svar = Selvstendig_n??ringsdrivende_doct[Selvstendig_n??ringsdrivende]
    Barn_under_18_svar = Barn_under_18_doct[Barn_under_18]
    eiendomsomr??de_svar = eiendomsomr??de_doct[Eiendomsomr??det]
    kreditt_svar = Kreditt_doct[Kreditt]
except:
    pass


try:
    X = [Kj??nn_svar, Gift_svar, Barn_under_18_svar, Utdannelse_svar, Selvstendig_n??ringsdrivende_svar, 
         Inntekt, Meds??kerinntekt, L??n, L??netid, kreditt_svar, eiendomsomr??de_svar]
    X = np.array(X)
except:
    pass  


if st.button("Send s??knad"):
    prediction = predict(X.reshape(1, -1), model)
    if prediction == 1:
        st.write(f"Basert p?? det du har oppgitt, s?? er l??nes??knaden din Godkjent. Derfor kan du l??ne: {L??n*1000} USD i {L??netid} mnd")
    else:
        st.write(f"Basert p?? det du har oppgitt, s?? er l??nes??knaden din ikke Godkjent. Derfor kan du ikke l??ne: {L??n*1000} USD i {L??netid} mnd")
    


st.write("Av Anish Thangalingam og Gurubaran Rajeshwaran")

st.write("Dataen er tatt fra: https://nmbu.instructure.com/courses/8010/files/folder/Prosjektoppgave/Automatisering%20av%20behandling%20av%20l%C3%A5nes%C3%B8knader")    
    
    
    