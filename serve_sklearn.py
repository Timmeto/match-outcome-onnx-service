from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd

# FastAPI-Instanz
app = FastAPI()

# Modell laden
pipe = load("models/match_pipeline.joblib")

# Request-Schema
class Row(BaseModel):
    data: dict  # Keys = Spaltennamen aus matches.csv (ohne 'win')

# Endpoint
@app.post("/api/winprob")
def winprob(r: Row):
    df = pd.DataFrame([r.data])
    prob = float(pipe.predict_proba(df)[0, 1])
    return {"winProb": prob}
