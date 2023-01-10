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

gender = st.radio("Kjønn",("Male", "Female"), key=1)

Gift = st.selectbox("Gift", ["Yes", "No"],key=1)

Selvstendig = st.selectbox("Selvsendig drivende", ["Yes", "No"], key=2,
                           help="Hvordan driver henter du inntekt?")

Utdanning = st.selectbox("Utdanning", ["Yes", "No"], key=3,
                         help="Utdanning høgre enn vidergåande.")

Barn = st.number_input("Barn under 18 år", min_value=0, max_value=100, value=0, step=1, key=2)

Eiendom = st.selectbox("Eiendomsområde", ["Urban", "Semiurban", "Landlig"], key=4,
                       help="Hvilke eigendomsområder er du fra av de tre alternativene?")

Kredit_hist = st.number_input("Kredit historie", min_value=0.0, max_value=1.0, step=0.05, value=0.5, key=3,
                              help="Her snakker vi om kreditt verdien.")

#Inntekt = st.slider("Inntekt",0, 999999999, 5000)

with st.sidebar:
    st.subheader('About')
    st.markdown('This dashboard is made by Joel, Dino and Trish, using **Streamlit**')
    # st.markdown("""#En lånesøknad er en formell anmodning om å få låne penger fra en bank eller annen kredittinstitusjon.
    #             Søknaden skal inneholde informasjon om den personen som søker lån, for eksempel inntekt, formue,
    #              gjeld og personlig informasjon. Søknaden skal også inneholde informasjon om hva lånet skal brukes
    #              til, samt den ønskede lånebeløpet og løpetiden.
    #
    #              Det er viktig at all informasjonen som gis i en lånesøknad er korrekt og oppdatert, siden bankene
    #              eller kredittinstitusjonene vil bruke denne informasjonen for å vurdere søkerens kredittverdighet
    #              og avgjøre om lånesøknaden skal godkjennes eller ikke.
    #
    #              Det er også viktig å sjekke og sammenligne ulike tilbud og lånebetingelser fra forskjellige
    #              institusjoner før en avgjørelse tas. Så i korte trekk så er en lånesøknad en formell anmodning
    #              om lån fra en institusjon som skal inneholde informasjonen om deg som person, dine økonomiske
    #              forhold og hva lånet skal brukes til.
    #              """)

st.write("You selected:",
         ("Alder", alder),
         ("Kjønn", gender),
         ("Selvstendig drivende", Selvstendig),
         ("Utdanning", Utdanning),
         ("Barn under 18år", Barn),
         ("Eiendomsområde", Eiendom),
         ("Kredit historie", Kredit_hist)
         )
# st.write(m.run(window=15))
# test
