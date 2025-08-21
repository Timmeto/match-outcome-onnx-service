import requests, json
payload = {
  "data": {
    "map": "SummonersRift",
    "patch": "14.9",
    "queue": "ranked",
    "top_champ": "Garen",
    "jungle_champ": "LeeSin",
    "mid_champ": "Ahri",
    "bot_champ": "Caitlyn",
    "supp_champ": "Lux",
    "avg_mmr": 1500,
    "avg_kda": 2.5
  }
}
r = requests.post("http://localhost:8080/api/winprob", json=payload, timeout=10)
print(r.status_code, r.json())
