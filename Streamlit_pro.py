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
         # Dataframe 
    """)
    
with st.sidebar:
    st.subheader('About')
    st.markdown('This dashboard is made by Joel, Dino and Trish, using **Streamlit**')
    st.markdown('This site wil calculate the consumer avaliablity to receive an "lånesøknad')
    
st.write(
pd.DataFrame({
    'A': [1, 5, 9, 7],
    'B': [3, 2, 4, 8]
  })
)
#st.write(m.run(window=15))
# test