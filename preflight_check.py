import sys
import os

print("=" * 65)
print("PRE-FLIGHT SYSTEM VERIFICATION")
print("=" * 65)

# 1. Module Imports Check
print("\n[1/3] Checking Core Imports...")
try:
    from core.vault_pool import get_vault_pool_model, VaultPoolChatModel
    print("  ✅ core.vault_pool: OK")
except Exception as e:
    print(f"  ❌ core.vault_pool import error: {e}")
    sys.exit(1)

try:
    from core import llm_provider
    print("  ✅ core.llm_provider: OK")
except Exception as e:
    print(f"  ❌ core.llm_provider import error: {e}")
    sys.exit(1)

try:
    from core import roman_urdu_translator
    print("  ✅ core.roman_urdu_translator: OK")
except Exception as e:
    print(f"  ❌ core.roman_urdu_translator import error: {e}")
    sys.exit(1)

# 2. Hook Resolution Check
print("\n[2/3] Checking 'master-auto-pool' Hook Integration...")
try:
    chat_model = llm_provider.get_chat_model(model="master-auto-pool")
    if isinstance(chat_model, VaultPoolChatModel):
        print(f"  ✅ Hook Resolution: OK (Returned {type(chat_model).__name__})")
    else:
        print(f"  ⚠️ Warning: Returned model type is {type(chat_model).__name__}")
except Exception as e:
    print(f"  ❌ Hook resolution failed: {e}")
    sys.exit(1)

# 3. End-to-End Live Ping
print("\n[3/3] Testing Live Auto-Pool Request...")
try:
    response = chat_model.invoke("Reply with exactly: 'SYSTEM_READY'")
    print(f"  ✅ Live Response: {response.content.strip()}")
except Exception as e:
    print(f"  ❌ Live call error: {e}")
    sys.exit(1)

print("\n" + "=" * 65)
print("🎉 ALL PRE-FLIGHT CHECKS PASSED — READY FOR STREAMLIT!")
print("=" * 65)