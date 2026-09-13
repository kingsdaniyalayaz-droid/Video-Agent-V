"""
Multi-Tier Auto-Pool Manager with Smart Cooldown
Strict Sequence: Tier 1 (Gemini) -> Tier 2 (Groq) -> Tier 3 (OpenRouter)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_GLOBAL_SLOT_COOLDOWNS: dict[int, float] = {}

def _resolve_vault_path() -> str:
    candidates = [
        "data/api_keys_vault.json",
        "api_keys_vault.json",
        os.path.join(os.path.dirname(__file__), "..", "data", "api_keys_vault.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "data/api_keys_vault.json"

class VaultPoolChatModel(BaseChatModel):
    temperature: float = 0.2
    timeout: float = 25.0  # Fast failover (25s max)
    cooldown_seconds: float = 90.0  # Failed slot ko 90s tak skip karega
    
    _tier1_slots: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _tier2_slots: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _tier3_slots: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, temperature: float = 0.2, timeout: float = 25.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.timeout = timeout
        self._load_slots()

    @property
    def _llm_type(self) -> str:
        return "vault-pool-chat-model"

    def _load_slots(self) -> None:
        vault_path = _resolve_vault_path()
        if not os.path.exists(vault_path):
            return

        with open(vault_path, "r", encoding="utf-8") as f:
            vault = json.load(f)

        slots = vault.get("slots", [])
        t1, t2, t3 = [], [], []

        for s in slots:
            api_key = s.get("api_key", "").strip()
            if not api_key or api_key.startswith("PASTE_"):
                continue

            slot_num = s.get("slot", 0)
            model = s.get("model", "")
            base_url = s.get("base_url", "").rstrip("/")
            tier = s.get("tier", "")

            client = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=self.temperature,
                timeout=self.timeout,
                max_retries=0  # No internal hanging retries
            )
            entry = {
                "slot": slot_num,
                "tier": tier,
                "model": model,
                "client": client,
            }

            provider_type = s.get("provider", "").lower()
            tier_name = s.get("tier", "").lower()

            if "gemini" in tier_name or "google" in provider_type or (1 <= slot_num <= 6):
                t1.append(entry)
            elif "groq" in provider_type or "groq" in tier_name or (7 <= slot_num <= 12) or (19 <= slot_num <= 24):
                t2.append(entry)
            elif "openrouter" in provider_type or "openrouter" in tier_name or (13 <= slot_num <= 18):
                t3.append(entry)

        self._tier1_slots = sorted(t1, key=lambda x: x["slot"])
        self._tier2_slots = sorted(t2, key=lambda x: x["slot"])
        self._tier3_slots = sorted(t3, key=lambda x: x["slot"])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any
    ) -> ChatResult:
        now = time.time()
        tiers = [
            ("Tier 1 - Google Gemini", self._tier1_slots),
            ("Tier 2 - Groq", self._tier2_slots),
            ("Tier 3 - OpenRouter", self._tier3_slots),
        ]

        last_error = None
        for tier_name, slot_list in tiers:
            for item in slot_list:
                slot_id = item["slot"]
                model_name = item["model"]
                client = item["client"]

                # Check agar ye slot cooldown mein hai to bina time zaya kiye skip karein
                cooldown_until = _GLOBAL_SLOT_COOLDOWNS.get(slot_id, 0)
                if now < cooldown_until:
                    continue

                try:
                    result = client._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    return result
                except Exception as err:
                    last_error = err
                    # Slot ko cooldown mein daal dein
                    _GLOBAL_SLOT_COOLDOWNS[slot_id] = time.time() + self.cooldown_seconds
                    print(f"⚠️ [{tier_name}] Slot {slot_id:02d} failed ({str(err)[:45]}). Cooldown {int(self.cooldown_seconds)}s -> Next slot...")

        # Agar saare slots cooldown mein hon to cooldown clear karke ek aur attempt karein
        _GLOBAL_SLOT_COOLDOWNS.clear()
        raise RuntimeError(f"All slots exhausted across all tiers. Last error: {last_error}")

def get_vault_pool_model(temperature: float = 0.2, **kwargs: Any) -> BaseChatModel:
    return VaultPoolChatModel(temperature=temperature, **kwargs)