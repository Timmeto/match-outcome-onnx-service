import os, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

# --- Daten laden & Header säubern ---
df = pd.read_csv("data/matches_dummy.csv")          # <— Pfad bei dir
df.columns = df.columns.str.strip()           # führende/trailing Spaces entfernen

# Ziel/Features
y = df["win"].astype(int)
X = df.drop(columns=["win"])

# Spaltentypen
cat = [c for c in X.columns if X[c].dtype == "object"]
num = [c for c in X.columns if c not in cat]

# Preprocessing + Modell
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
    ("num", StandardScaler(), num)
], verbose_feature_names_out=False)

pipe = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=300))
])

# Split (stratify wenn möglich)
strat = y if (y.nunique() == 2 and (y.value_counts() >= 2).all()) else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)

# Train
pipe.fit(Xtr, ytr)

# Eval (nur wenn beide Klassen im Test)
if yte.nunique() == 2:
    p = pipe.predict_proba(Xte)[:, 1]
    print("ROC-AUC:", roc_auc_score(yte, p))
else:
    print("ROC-AUC übersprungen (Testset hat nur eine Klasse).")

# --- Speichern (sklearn-Pipeline) ---
os.makedirs("models", exist_ok=True)
dump(pipe, "models/match_pipeline.joblib")
with open("models/feature_schema.json", "w") as f:
    json.dump({"categorical": cat, "numerical": num}, f, indent=2)

print("Saved -> models/match_pipeline.joblib")
print("Schema -> models/feature_schema.json")
