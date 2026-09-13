import json
import os
import requests

VAULT_FILE = "api_keys_vault.json"

if not os.path.exists(VAULT_FILE):
    print(f"File '{VAULT_FILE}' nahi mili. Check karein ke file project root mein hai.")
    exit(1)

with open(VAULT_FILE, "r", encoding="utf-8") as f:
    vault = json.load(f)

slots = vault.get("slots", [])
print("=" * 75)
print(f"TESTING {len(slots)} SLOTS FROM {VAULT_FILE}")
print("=" * 75)

for s in slots:
    slot_num = s.get("slot")
    tier = s.get("tier")
    model = s.get("model")
    base_url = s.get("base_url", "").rstrip("/")
    api_key = s.get("api_key", "").strip()

    # Agar placeholder ho to skip karein
    if api_key.startswith("PASTE_") or not api_key:
        print(f"Slot {slot_num:02d} | [{tier}] | ⚠️ UNCONFIGURED (Key paste nahi hui)")
        continue

    # OpenAI-compatible /chat/completions endpoint
    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        if res.status_code == 200:
            print(f"Slot {slot_num:02d} | [{tier}] | ✅ SUCCESS (200 OK) | Model: {model}")
        else:
            err_snippet = res.text.replace("\n", " ")[:90]
            print(f"Slot {slot_num:02d} | [{tier}] | ❌ FAIL ({res.status_code}) | Error: {err_snippet}")
    except Exception as exc:
        print(f"Slot {slot_num:02d} | [{tier}] | ❌ NETWORK ERROR | {str(exc)[:60]}")

print("=" * 75)