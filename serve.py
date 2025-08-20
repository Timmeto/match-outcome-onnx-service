from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import pandas as pd
import numpy as np

app = FastAPI()

# ONNX-Session laden
sess = ort.InferenceSession("models/match_baseline.onnx", providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
output_name = sess.get_outputs()[0].name

class Row(BaseModel):
    data: dict  # {spaltenname: wert} – exakt wie in deiner matches.csv (ohne 'win')

def _predict_one(row: dict) -> float:
    """
    Robust: Probiert mehrere Zufuhr-Formate, da skl2onnx-Exports je nach Pipeline
    1 Input (2D-Tensor) ODER mehrere Inputs (eine Spalte pro Input) erzeugen können.
    """
    # 1) Versuch: alle Spalten als einzelne Inputs (wenn das Modell mehrere Inputs hat)
    if len(inputs) > 1:
        ort_inputs = {}
        for inp in inputs:
            name = inp.name
            if name not in row:
                raise ValueError(f"Fehlender Key im Request: '{name}'")
            val = row[name]
            # Dtype heuristisch bestimmen
            dtype = np.float32
            if "string" in inp.type or isinstance(val, str):
                dtype = object  # onnxruntime akzeptiert object für String
            arr = np.array([[val]], dtype=dtype)
            ort_inputs[name] = arr
        pred = sess.run([output_name], ort_inputs)[0]
    else:
        # 2) Versuch: ein einzelner Input – DataFrame -> 2D-Array (float32)
        df = pd.DataFrame([row])
        arr = df.values.astype(np.float32)
        pred = sess.run([output_name], {inputs[0].name: arr})[0]

    # Wahrscheinlichkeit auslesen (Binary-Klassifikation)
    if pred.ndim == 2 and pred.shape[1] == 2:
        return float(pred[0][1])  # Positivklasse
    return float(pred[0] if np.ndim(pred[0]) == 0 else pred[0][0])

@app.post("/api/winprob")
def winprob(r: Row):
    prob = _predict_one(r.data)
    return {"winProb": prob}
