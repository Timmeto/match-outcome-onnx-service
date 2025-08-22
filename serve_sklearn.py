from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from joblib import load
import pandas as pd, json, os, traceback

app = FastAPI()

# Modell laden
PIPE_PATH = "models/match_pipeline.joblib"
if not os.path.exists(PIPE_PATH):
    raise RuntimeError(f"Model not found: {PIPE_PATH}. Run training first.")
pipe = load(PIPE_PATH)

# Schema laden (falls vorhanden)
SCHEMA_PATH = "models/feature_schema.json"
schema = {"categorical": [], "numerical": []}
if os.path.exists(SCHEMA_PATH):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
expected_keys = schema.get("categorical", []) + schema.get("numerical", [])

class Row(BaseModel):
    data: dict  # Keys = Spaltennamen deiner matches.csv (ohne 'win')

# Globale Fehler -> JSON (kein HTML/Plaintext)
@app.middleware("http")
async def error_wrapper(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": type(e).__name__,
                "path": request.url.path,
                "trace": traceback.format_exc().splitlines()[-5:],  # letzte Zeilen
            },
        )

@app.get("/api/ping")
def ping():
    return {"ok": True}

@app.get("/api/schema")
def get_schema():
    return {"expected_keys": expected_keys, **schema}

@app.post("/api/winprob")
def winprob(r: Row):
    # Optional: Eingabeschlüssel prüfen, wenn schema existiert
    if expected_keys:
        missing = [k for k in expected_keys if k not in r.data]
        if missing:
            return JSONResponse(status_code=422, content={
                "error": "missing keys",
                "missing": missing,
                "expected": expected_keys
            })

    df = pd.DataFrame([r.data])
    prob = float(pipe.predict_proba(df)[0, 1])
    return {"winProb": prob}
