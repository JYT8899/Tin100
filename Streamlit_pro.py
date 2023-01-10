#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:51:21 2023

@author: fbi
"""

import streamlit as st
import pandas as pd

st.title("""
          Lånesøknad
    """)

st.write("""
          Spørreundersøkelse
    """)

alder = st.number_input("Alder", min_value=0, max_value=100, value=30, step=1, key=1)

gender = st.radio("Kjønn",("Male", "Female"), key=2)

Gift = st.selectbox("Gift", ["Yes", "No"],key=3)

Selvstendig = st.selectbox("Selvsendig drivende", ["Yes", "No"], key=4,
                           help="Hvordan driver, henter du inntekt?")

Utdanning = st.selectbox("Utdanning", ["Yes", "No"], key=5,
                         help="Utdanning høgre enn vidergåande.")

Barn = st.number_input("Barn under 18 år", min_value=0, max_value=100, value=0, step=1, key=6)

Eiendom = st.selectbox("Eiendomsområde", ["Urban", "Semiurban", "Landlig"], key=7,
                       help="Hvilke eigendomsområder er du fra av de tre alternativene?")

Kredit_hist = st.slider("Kredit historie", 0.0, 1.0, 0.05)

Inntekt = st.number_input("Inntekt", min_value=0.0, max_value=10000000.0, step=5000.0, value=45000.0, key=8,
                              help="Her snakker vi om bruttoinntekt (inntekt før skatt)")

Medsokerinntekt = st.number_input("Medsøkerinntekt", min_value=0.0, max_value=10000000.0, step=5000.0, value=25000.0, key=9,
                              help="Medsøkerinntekt er inntekten til en person som søker sammen med deg om å få et "
                                   " lån eller annen form for finansiering. Dette kan være en ektefelle, samboer eller"
                                   " noen annen form for partner.")

Laan = st.number_input("Lån (antall 1000)", min_value=0.0, max_value=10000000.0, step=5000.0, value=25000.0, key=10,
                              help="Hvor mye vil du låne?")

mnd= st.slider("Låne lengde (antall måned)", 0, 360, 1, help="Hvor lang tid vil du låne?", key=11)

# NB! må få fikste if setningen skikkerlig når vi blir ferdig med ML delen. 
if st.button('Send søknad'):
    pred = pred(X.reshape(1,-1), model)
    if prediction == 1:
        st.write('Takk for søknaden, din søknad er akseptert. Du kan låne: {Lån * 1000} USD i {Lånetid} måneder')
    else:
        st.write('Takk for søknaden, din søknad er dessverre ikke akseptert. Du kan ikkj låne: {Lån * 1000} USD i {Lånetid} måneder')

with st.sidebar:
    st.subheader('Om siden')
    st.markdown('En lånesøknad er en formell anmodning om å få låne penger fra en bank eller annen kredittinstitusjon.'
                ' Søknaden skal inneholde informasjon om den personen som søker lån, for eksempel inntekt, formue,'
                ' gjeld og personlig informasjon. Søknaden skal også inneholde informasjon om hva lånet skal brukes'
                ' til, samt den ønskede lånebeløpet og løpetiden.')

    st.markdown('Det er viktig at all informasjonen som gis i en lånesøknad er korrekt og oppdatert, siden bankene'
                ' eller kredittinstitusjonene vil bruke denne informasjonen for å vurdere søkerens kredittverdighet'
                ' og avgjøre om lånesøknaden skal godkjennes eller ikke.')

    st.markdown('Det er også viktig å sjekke og sammenligne ulike tilbud og lånebetingelser fra forskjellige'
                ' institusjoner før en avgjørelse tas. Så i korte trekk så er en lånesøknad en formell anmodning'
                ' om lån fra en institusjon som skal inneholde informasjonen om deg som person, dine økonomiske'
                ' forhold og hva lånet skal brukes til.')
    st.markdown('This dashboard is made by Joel, Dino and Trish, using **Streamlit**')

st.write("Du har valgt:",
         ("Alder", alder),
         ("Kjønn", gender),
         ("Selvstendig drivende", Selvstendig),
         ("Utdanning", Utdanning),
         ("Barn under 18år", Barn),
         ("Eiendomsområde", Eiendom),
         ("Kredit historie", Kredit_hist),
         ("Inntekt", Inntekt),
         ("Medsøkerinntekt", Medsokerinntekt),
         ("Lån", Laan),
         ("Låne lengde", mnd)
         )
# st.write(m.run(window=15))
# test
