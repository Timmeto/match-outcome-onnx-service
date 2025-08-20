import os, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Erwartet data/matches.csv mit Zielspalte 'win' (0/1)
df = pd.read_csv("data/matches.csv")
y = df["win"].astype(int)
X = df.drop(columns=["win"])

cat = [c for c in X.columns if X[c].dtype == "object"]
num = [c for c in X.columns if c not in cat]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat),
    ("num", StandardScaler(), num)
])
pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=300))])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(Xtr, ytr)
p = pipe.predict_proba(Xte)[:,1]
print("ROC-AUC:", roc_auc_score(yte, p))

os.makedirs("models", exist_ok=True)
feat_count = pipe[:-1].transform(Xtr[:1]).shape[1]
onnx = convert_sklearn(pipe, initial_types=[("input", FloatTensorType([None, feat_count]))])
open("models/match_baseline.onnx","wb").write(onnx.SerializeToString())
print("Export -> models/match_baseline.onnx")
