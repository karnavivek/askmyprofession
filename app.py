import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("AskMyProfession")
st.write("Choose profession from the dropbox & Ask any question related to this field!")
st.write("Try it! its like talking to an expert of that Profession!")

profession = st.selectbox(
    "Choose a Profession you'd like your answers from:",
    ("Operations Research Scientist", "Data Scientist", "Supply Chain Consultant")
)

if profession == "Operations Research Scientist":
    #run llm model which is fine-tuned with operations Research knowledge
    pass


elif profession == "Data Scientist":
    #run llm model which is fine-tuned with Data Science knowledge, it can be specific too
    pass

else:
    #run llm model which is fine-tuned with Supply Chain Consultant knowledge
    pass


