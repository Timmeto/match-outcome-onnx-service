import requests
payload = {
  "data": {
    "map": "SummonersRift",
    "queue": "ranked",
    "top_champ": "Garen",
    "jungle_champ": "LeeSin",
    "mid_champ": "Ahri",
    "bot_champ": "Caitlyn",
    "support_champ": "Lux",
    "patch": "14.9",
    "avg_mmr": 1500,
    "avg_kda": 2.5
  }  
}
r = requests.post("http://localhost:8080/api/winprob", json=payload, timeout=10)
print("STATUS:", r.status_code)
print("CT  :", r.headers.get("content.type"))
print("TEXT :", r.text[:500])
try:
    print("JSON :", r.json())
except Exception as e: 
    print("JSON-Error:", e)
