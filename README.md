# 🌾 Sustainable Agriculture Agentic System

### Predictive irrigation & crop-stress forecasting using real Indian dataset

This project demonstrates a **multi-agent AI system** that integrates soil, climate, and satellite indicators to generate **data-driven irrigation and crop health recommendations**.

---

## 🚀 Features
- **DataFusion Agent:** Combines soil + climate + NDVI to forecast 24h soil moisture.
- **Irrigation Agent:** Recommends water volume via fuzzy decision logic.
- **Pest Agent:** Estimates pest/disease risk from canopy stress.
- **TinyMLOps Service:** Simulates model drift monitoring and OTA update triggers.
- **Interactive Gradio UI:** Live zone selection and result display.

---

## 🧠 Tech Stack
Python, Pandas, Scikit-learn, Gradio, AsyncIO, Random Forests, TinyMLOps simulation.

---

## 🧩 How to Run
### In Google Colab
1. Upload `CropDataset-Enhanced.csv`
2. Run `Sustainable_Agriculture_Agentic_System.ipynb` step-by-step.
3. The Gradio UI opens inline (or via public link).

### Local (Python ≥3.10)
```bash
pip install -r requirements.txt
python app.py
