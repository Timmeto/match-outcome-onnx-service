import os, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skl2onnx import to_onnx

# --- Daten laden & Header säubern ---
df = pd.read_csv("data/matches_dummy.csv")
df.columns = df.columns.str.strip()   # <— entfernt führende/trailing Leerzeichen

# Ziel/Features
y = df["win"].astype(int)
X = df.drop(columns=["win"])

# Spalten-Typen bestimmen
cat = [c for c in X.columns if X[c].dtype == "object"]
num = [c for c in X.columns if c not in cat]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
    ("num", StandardScaler(), num)
])

pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=300))])

# Split (stratify nur wenn genug Klassen vorhanden)
strat = y if y.nunique() == 2 and all((y.value_counts() >= 2)) else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=42, stratify=strat)

pipe.fit(Xtr, ytr)

# AUC nur wenn beide Klassen im Test existieren
if len(np.unique(yte)) == 2:
    p = pipe.predict_proba(Xte)[:, 1]
    print("ROC-AUC:", roc_auc_score(yte, p))
else:
    print("ROC-AUC übersprungen (Testset hat nur eine Klasse).")

# --- ONNX Export: to_onnx mit DataFrame-Sample (behält Spaltennamen) ---
os.makedirs("models", exist_ok=True)
sample = Xtr.head(1)              # DataFrame mit Spaltennamen!
onnx_model = to_onnx(pipe, sample, target_opset=17, options={"zipmap": False})  # zipmap=False nicht nötig für simple LogReg
with open("models/match_baseline.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Export -> models/match_baseline.onnx")
