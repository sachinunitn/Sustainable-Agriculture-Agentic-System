import streamlit as st
import pandas as pd
import joblib
import asyncio
import nest_asyncio
from app import AgentCoordinator

nest_asyncio.apply()

st.set_page_config(page_title="🌾 Sustainable Agriculture AI", page_icon="🌾")
st.title("🌾 Sustainable Agriculture Agentic System")
st.markdown("**Predictive irrigation & crop-stress forecasting using Indian data.**")

# Load data and model ----------------------------------------------------------
@st.cache_resource
def load_data():
    return pd.read_csv("CropDataset-Enhanced.csv")

@st.cache_resource
def load_model():
    try:
        return joblib.load("soil_model.pkl")
    except Exception:
        return None

df = load_data()
rf = load_model()

if df is None or len(df) == 0:
    st.error("Dataset missing or empty. Ensure `CropDataset-Enhanced.csv` is in the repo root.")
else:
    zones = df["Address"].dropna().unique().tolist()
    selected_zone = st.selectbox("Select Zone", zones)

    if st.button("Run Prediction"):
        try:
            row = df[df["Address"].astype(str).str.contains(selected_zone, case=False, na=False)]
            if len(row) == 0:
                row = df.iloc[[0]]
            rec = {k: v for k, v in row.iloc[0].items() if isinstance(v, (int, float))}
            coord = AgentCoordinator(rf)

            f, i, p, m = asyncio.get_event_loop().run_until_complete(coord.run_pipeline(rec))

            st.success(f"💧 {i['volume_liters']} L water recommended")
            st.info(f"🌿 Stress Index {f['canopy_stress_index']}")
            st.warning(f"🐛 Pest Risk {p['pest_risk']}")
            st.caption(f"🧩 Drift {m['drift_score']} ({m['status']})")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
