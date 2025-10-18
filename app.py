"""
Sustainable Agriculture Agentic System
--------------------------------------
Predictive irrigation and crop-stress forecasting using multi-agent AI.
Author: [Your Name]
License: MIT
"""

import asyncio, random, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import gradio as gr

# ─── Load Dataset ───
DATA_PATH = "CropDataset-Enhanced.csv"
df = pd.read_csv(DATA_PATH)

# Clean % columns
for c in df.columns:
    if df[c].astype(str).str.contains("%").any():
        df[c] = df[c].astype(str).str.replace("%","",regex=False).astype(float)
df.fillna(df.median(numeric_only=True), inplace=True)

# ─── Train simple RandomForest ───
num_cols = df.select_dtypes(include=["number"]).columns.tolist()[:6]
df["Soil_Moisture"] = (
    0.3*df[num_cols[0]] + 0.25*df[num_cols[1]] + 0.2*df[num_cols[2]] -
    0.1*df[num_cols[3]] + np.random.normal(0,1,len(df))
)
X, y = df[num_cols], df["Soil_Moisture"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "soil_model.pkl")

# ─── Agents ───
class DataFusionAgent:
    def __init__(self, model): self.model = model
    async def process(self, rec):
        x = np.array([[rec[k] for k in list(rec.keys())[:6]]])
        base = self.model.predict(x)[0]
        forecast = [base + random.uniform(-1,1) for _ in range(24)]
        stress = round(1 - (np.mean(forecast)/100), 2)
        return {"soil_moisture_24h": forecast, "canopy_stress_index": stress}

class IrrigationAgent:
    async def process(self, zone, forecast):
        avg = np.mean(forecast)
        dose = 200 if avg < 30 else 0
        reason = f"Avg soil moisture {avg:.1f}% → {'needs irrigation' if dose>0 else 'no irrigation'}"
        return {"zone": zone, "volume_liters": dose, "reasoning": reason}

class PestSurveillanceAgent:
    async def process(self, stress):
        risk = "High" if stress > 0.7 else "Low"
        return {"pest_risk": risk}

class TinyMLOpsService:
    def __init__(self): self.version = "v1.0.0"
    async def monitor(self):
        drift = random.uniform(0,0.3)
        status = "retrain" if drift>0.15 else "stable"
        return {"model_version": self.version, "drift_score": round(drift,3), "status": status}

class AgentCoordinator:
    def __init__(self, model):
        self.dfuse=DataFusionAgent(model)
        self.irrig=IrrigationAgent()
        self.pest=PestSurveillanceAgent()
        self.mlops=TinyMLOpsService()
    async def run_pipeline(self, rec):
        f=await self.dfuse.process(rec)
        i=await self.irrig.process(rec.get("Address","Zone-A"),f["soil_moisture_24h"])
        p=await self.pest.process(f["canopy_stress_index"])
        m=await self.mlops.monitor()
        return f,i,p,m

# ─── Gradio UI ───
def run_zone(zone):
    row = df[df["Address"].astype(str).str.contains(zone,case=False,na=False)]
    if len(row)==0: row = df.iloc[[0]]
    rec = {k:v for k,v in row.iloc[0].items() if isinstance(v,(int,float))}
    coord = AgentCoordinator(rf)
    f,i,p,m = asyncio.run(coord.run_pipeline(rec))
    return f"💧 {i['volume_liters']}L | 🌿 Stress {f['canopy_stress_index']} | 🐛 Pest {p['pest_risk']} | Drift {m['drift_score']} ({m['status']})"

zones = df["Address"].dropna().unique().tolist()[:15]
ui = gr.Interface(fn=run_zone, inputs=gr.Dropdown(zones, label="Select Zone"), outputs="text",
                  title="🌾 Sustainable Agriculture AI System",
                  description="Predictive irrigation and crop-stress forecasting using Indian data.")
ui.launch()
