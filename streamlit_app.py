import streamlit as st
import pandas as pd
import joblib
from main_module import AgentCoordinator  # or adjust import path

st.title("🌾 Sustainable Agriculture AI System")
st.write("Predictive irrigation & crop-stress forecasting on Indian data.")

df = pd.read_csv("CropDataset-Enhanced.csv")
rf = joblib.load("soil_model.pkl")
zones = df["Address"].dropna().unique().tolist()

selected_zone = st.selectbox("Select Zone", zones)

if st.button("Run Prediction"):
    rec = df[df["Address"].str.contains(selected_zone, case=False, na=False)].iloc[0].to_dict()
    coord = AgentCoordinator(rf)
    result = coord.run_pipeline(rec)  # adjust if async
    st.success(result)
