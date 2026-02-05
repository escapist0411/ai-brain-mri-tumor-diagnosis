import streamlit as st
import os
import pandas as pd

st.title("📁 Report History")

folder = "reports/history"
os.makedirs(folder, exist_ok=True)

files = os.listdir(folder)

if files:
    df = pd.DataFrame(files, columns=["Report Files"])
    st.dataframe(df)
else:
    st.info("No reports available.")
