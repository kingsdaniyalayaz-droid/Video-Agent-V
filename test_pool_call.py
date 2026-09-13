from core.vault_pool import get_vault_pool_model

print("Testing Master Auto-Pool call...")
pool = get_vault_pool_model()
res = pool.invoke("Hello, reply with 3 words in Roman Urdu.")
print(f"\n✅ Pool Response: {res.content}")