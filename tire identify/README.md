# Tire Print → Car Model Dashboard

A modern Streamlit dashboard where you upload a tire print image and get a predicted car model with confidence. The app includes a crisp preprocessing preview (edges/contrast) and lets you switch between a vertical card layout or a split horizontal view.

## Features
- Image upload (PNG/JPG/WebP)
- Preprocessing preview (Sobel + Canny)
- Deterministic mock prediction of car model (replaceable with real ML model)
- Attractive UI with two layout modes

## Quick Start

### 1) Activate your venv (Windows PowerShell)
```powershell
.\venv\Scripts\activate
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

### 3) Run the app
```powershell
streamlit run app.py
```

Open the displayed local URL in your browser (usually http://localhost:8501).

## Plug in a Real Model
Replace the logic in `predictor.py`'s `predict_car_model()` with your trained model:
- Load your model (e.g., from a `.pkl`, `.onnx`, or `.pt` file)
- Convert the uploaded PIL image to the input tensor/array your model expects
- Return `(model_name, confidence)`

Keep the preprocessing utility `preprocess_tire_print()` for the visual preview or adjust as needed.

## Notes
- The current predictor is a mock to demonstrate the end-to-end flow. It is deterministic for the same image and produces a reasonable confidence signal from image statistics.
- If you add heavier libraries (OpenCV, Torch, etc.), update `requirements.txt` accordingly.
