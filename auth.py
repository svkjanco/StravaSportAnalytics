from urllib.parse import urlparse, parse_qs
import config
from src import strava_client as sc

print("\n1) Otvor tuto URL v prehliadaci, prihlas sa a klikni Authorize:\n")
print("   " + sc.authorize_url(config.CLIENT_ID))
print("\n   Presmeruje ta na http://localhost/exchange_token?...&code=XXXX...")
print("   Stranka da 404 - to je OK. Potrebujes len hodnotu code.\n")

raw = input("2) Sem vloz code (alebo cely localhost URL) a stlac Enter: ").strip()
if "code=" in raw:
    raw = parse_qs(urlparse(raw).query).get("code", [raw])[0]

tok = sc.exchange_code(config.CLIENT_ID, config.CLIENT_SECRET, raw)
print("\nOK - token ulozeny do cache/token.json")
print("Scope:", tok.get("scope"))
