import os
import sys

sys.path.insert(0, r"c:\Projects\Video_Agent\Video_Agent")

from core.llm_provider import llm_identity, LLMConfig
import core.llm_provider as llm_provider

# set runtime config
llm_provider._runtime_config = LLMConfig(provider="groq", model="openai/gpt-oss-120b")

# Simulate _print_provider_status for Translator with no overrides
def test_translator():
    # In translate_to_roman_urdu, _get_llm is called with model=MISTRAL_MODEL
    # then _create_provider_client(model=MISTRAL_MODEL, ...)
    # which has model_override=None, provider_override=None
    
    model_override = None
    provider_override = None
    model = "mistral-small-latest"
    
    runtime_config = llm_provider._runtime_config
    passed_model = model_override
    if runtime_config is None and model_override is None:
        passed_model = model
        
    p, m, b = llm_identity(model=passed_model, provider=provider_override)
    print(f"Translator Identity -> Provider: {p}, Model: {m}")
    assert p == "groq"
    assert m == "openai/gpt-oss-120b"

def test_reviewer():
    # In get_reviewer_llm with dedicated provider but no model
    # model_override = None, provider_override = "openrouter"
    # model = MISTRAL_REVIEWER_MODEL ("mistral-small-latest")
    
    model_override = None
    provider_override = "openrouter"
    model = "mistral-small-latest"
    
    runtime_config = llm_provider._runtime_config
    passed_model = model_override
    if runtime_config is None and model_override is None:
        passed_model = model
        
    p, m, b = llm_identity(model=passed_model, provider=provider_override)
    print(f"Reviewer Identity -> Provider: {p}, Model: {m}")
    assert p == "openrouter"
    assert m == "openai/gpt-oss-120b"

def test_reviewer_both():
    model_override = "nex-agi/nex-n2.5-mini:free"
    provider_override = "openrouter"
    model = "mistral-small-latest"
    
    runtime_config = llm_provider._runtime_config
    passed_model = model_override
    if runtime_config is None and model_override is None:
        passed_model = model
        
    p, m, b = llm_identity(model=passed_model, provider=provider_override)
    print(f"Reviewer Identity (both) -> Provider: {p}, Model: {m}")
    assert p == "openrouter"
    assert m == "nex-agi/nex-n2.5-mini:free"

test_translator()
test_reviewer()
test_reviewer_both()
print("All passed")
