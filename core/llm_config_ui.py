"""Streamlit UI for the centralized universal LLM configuration layer.

This module owns provider presentation only.  Provider behavior, defaults,
validation, persistence, and adapter selection remain in ``llm_provider``.
Model is intentionally a free-text input: the selected provider/API is the
authority that validates whether a model exists.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from .llm_provider import (
    LLMConfig,
    _default_base_url_for,
    configure_llm,
    get_config_store,
    provider_alias,
    provider_label,
    supported_providers,
    validate_configuration,
)


PROVIDER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Auto Detect", ("auto",)),
    ("Direct Providers", ("openai", "google", "mistral", "anthropic", "groq", "nvidia")),
    ("Gateways / Aggregators", ("openrouter", "tokenrouter", "together", "deepinfra")),
    ("Local / Self Hosted", ("ollama", "lm_studio", "vllm")),
    ("Generic", ("openai_compatible",)),
)


def _provider_options() -> list[tuple[str, str]]:
    """Return only selectable provider entries; group headings are UI-only."""
    labels = {"auto": "Auto Detect"}
    options: list[tuple[str, str]] = []
    registered = set(supported_providers())
    for _group, keys in PROVIDER_GROUPS:
        for key in keys:
            if key == "auto" or key in registered:
                options.append((labels.get(key, provider_label(key)), key))
    return options


def _normalize_url(value: str | None) -> str:
    return (value or "").strip().rstrip("/").lower()


def _sync_provider_base_url(provider_key: str | None) -> None:
    """Update a default URL only when the user has not customized it.

    Streamlit reruns the script after every widget interaction.  The previous
    canonical provider is therefore kept in session state so a provider switch
    can distinguish a stale registry default from a URL manually entered by
    the user.
    """
    current_provider = provider_key or "auto"
    previous_provider = st.session_state.get("ai_cfg_previous_provider")
    current_url = str(st.session_state.get("ai_cfg_base_url") or "").strip()

    if previous_provider is None:
        if not current_url:
            initial_default = _default_base_url_for(provider_key) if provider_key else None
            if initial_default:
                st.session_state["ai_cfg_base_url"] = initial_default
    elif current_provider != previous_provider:
        previous_default = _default_base_url_for(None if previous_provider == "auto" else previous_provider)
        new_default = _default_base_url_for(provider_key) if provider_key else None
        if previous_default and _normalize_url(current_url) == _normalize_url(previous_default):
            # The URL is still a registry default, not a user customization.
            # Replace it for another known provider, or clear it when the new
            # provider intentionally has no default endpoint.
            st.session_state["ai_cfg_base_url"] = new_default or ""

    st.session_state["ai_cfg_previous_provider"] = current_provider


def build_config_from_form(values: dict[str, Any]) -> LLMConfig:
    """Normalize UI values through the one provider validator."""
    selected = str(values.get("provider_label") or "Auto Detect").strip()
    provider = None if selected.lower() in {"auto detect", "auto", "autodetect"} else provider_alias(selected)
    return validate_configuration(
        config_name=str(values.get("config_name") or "").strip() or None,
        provider=provider,
        model=str(values.get("model") or "").strip(),
        api_key=str(values.get("api_key") or "").strip(),
        base_url=str(values.get("base_url") or "").strip() or None,
        temperature=float(values.get("temperature", 0.2)),
    )


def _selected_label(canonical: str | None) -> str:
    if not canonical:
        return "Auto Detect"
    return provider_label(canonical)


def _set_config_notification(notification: dict[str, Any]) -> None:
    """Store safe configuration-result metadata for the application shell."""
    st.session_state["ai_config_notification"] = notification
    st.session_state["ai_config_notification_time"] = time.time()


def render_ai_model_settings() -> None:
    """Render and activate the persisted multi-provider configuration UI."""
    store = get_config_store()
    active_name = store.active_name()
    active = store.get(active_name) if active_name else None

    with st.expander("AI Model Settings", expanded=active is None):
        names = store.names()
        if names:
            selected_name = st.selectbox(
                "Saved Configuration",
                ["(New configuration)", *names],
                index=(names.index(active_name) + 1 if active_name in names else 0),
                key="ai_saved_config",
            )
            if selected_name != "(New configuration)" and st.button("Load", key="ai_load_config"):
                loaded = store.activate(selected_name)
                # Widget state survives reruns independently of the provider
                # store.  Synchronize every form field before rerunning so a
                # loaded configuration cannot display stale values from the
                # previously active configuration.
                st.session_state["ai_cfg_name"] = selected_name
                st.session_state["ai_cfg_provider"] = _selected_label(loaded.provider)
                st.session_state["ai_cfg_model"] = loaded.model
                st.session_state["ai_cfg_key"] = loaded.api_key
                st.session_state["ai_cfg_base_url"] = loaded.base_url or ""
                st.session_state["ai_cfg_temperature"] = float(loaded.temperature)
                st.session_state["ai_cfg_previous_provider"] = loaded.provider or "auto"
                st.rerun()

        default = active
        provider = _selected_label(default.provider if default else None)
        model = default.model if default else ""
        api_key = default.api_key if default else ""
        base_url = default.base_url if default else ""
        temperature = float(default.temperature if default else 0.2)
        config_name = active_name or (default.config_name if default else "") or ""

        # Seed the widget state from the active saved configuration before
        # provider-switch synchronization runs.  This preserves a saved
        # custom Base URL on a fresh Streamlit session instead of allowing the
        # provider's default URL to be inserted first.
        if "ai_cfg_base_url" not in st.session_state:
            st.session_state["ai_cfg_base_url"] = base_url or ""

        st.text_input("Configuration Name", value=config_name, key="ai_cfg_name")
        options = _provider_options()
        labels = [label for label, _ in options]
        selected_index = labels.index(provider) if provider in labels else 0
        chosen_label = st.selectbox("Provider", labels, index=selected_index, key="ai_cfg_provider")
        chosen_key = dict(options).get(chosen_label)
        _sync_provider_base_url(chosen_key)
        st.text_input("Model Name", value=model, key="ai_cfg_model", help="Enter the exact API model identifier.")
        st.text_input("API Key", value=api_key, type="password", key="ai_cfg_key")
        st.text_input("Base URL", key="ai_cfg_base_url")
        st.number_input("Temperature", min_value=0.0, max_value=2.0, value=temperature, step=0.1, key="ai_cfg_temperature")

        if st.button("Save and Activate", type="primary", key="ai_activate_config"):
            values = {
                "config_name": st.session_state.get("ai_cfg_name"),
                "provider_label": chosen_label,
                "model": st.session_state.get("ai_cfg_model"),
                "api_key": st.session_state.get("ai_cfg_key"),
                "base_url": st.session_state.get("ai_cfg_base_url"),
                "temperature": st.session_state.get("ai_cfg_temperature", 0.2),
            }
            try:
                config = build_config_from_form(values)
                name = store.save(config, name=config.config_name)
                store.activate(name)
                st.session_state["ai_cfg_previous_provider"] = config.provider or "auto"
                st.session_state["ai_model_settings"] = {
                "active": True,
                "config_name": name,
                "provider": config.provider,
                "model": config.model,
                "api_key": config.api_key,
                "base_url": config.base_url,
                "temperature": config.temperature,
                }
                _set_config_notification({
                    "type": "success",
                    "title": "AI Model Configuration Successfully Activated",
                    "config_name": name,
                    "provider": provider_label(config.provider),
                    "model": config.model,
                    "base_url": config.base_url or "",
                    "api_key_configured": bool(config.api_key),
                    "status": "Successfully Activated",
                })
            except Exception as exc:  # noqa: BLE001 - surface safe UI feedback
                safe_error = str(exc) or "Configuration could not be activated."
                submitted_key = str(values.get("api_key") or "")
                if submitted_key:
                    safe_error = safe_error.replace(submitted_key, "[REDACTED]")
                _set_config_notification({
                    "type": "error",
                    "title": "AI Model Configuration Failed",
                    "config_name": str(values.get("config_name") or "") or None,
                    "provider": str(values.get("provider_label") or "Auto Detect"),
                    "model": str(values.get("model") or "") or None,
                    "error": safe_error,
                    "status": "Activation Failed",
                })
            st.rerun()
