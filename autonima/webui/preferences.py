"""Global (non-secret) UI preferences for the Autonima web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PREFERENCES_KEY_MODELS = "preferred_models"
PREFERENCES_KEY_DEFAULT_MODEL = "default_model"


class PreferencesManager:
    """Manage UI preferences stored in ~/.autonima-ui.json."""

    def __init__(self, preferences_path: Path | None = None):
        self.preferences_path = preferences_path or (Path.home() / ".autonima-ui.json")

    def _normalize_models(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []

        cleaned: List[str] = []
        seen = set()
        for value in values:
            if not isinstance(value, str):
                continue
            model = value.strip()
            if not model or model in seen:
                continue
            seen.add(model)
            cleaned.append(model)
        return cleaned

    def _normalize_default_model(self, value: Any, allowed_models: List[str]) -> str:
        if not isinstance(value, str):
            return ""
        model = value.strip()
        if not model:
            return ""
        if model not in allowed_models:
            return ""
        return model

    def _normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        models = self._normalize_models(raw.get(PREFERENCES_KEY_MODELS, []))
        return {
            PREFERENCES_KEY_MODELS: models,
            PREFERENCES_KEY_DEFAULT_MODEL: self._normalize_default_model(
                raw.get(PREFERENCES_KEY_DEFAULT_MODEL, ""),
                models,
            ),
        }

    def load(self) -> Dict[str, Any]:
        if not self.preferences_path.exists():
            return {
                PREFERENCES_KEY_MODELS: [],
                PREFERENCES_KEY_DEFAULT_MODEL: "",
            }

        try:
            payload = json.loads(self.preferences_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                PREFERENCES_KEY_MODELS: [],
                PREFERENCES_KEY_DEFAULT_MODEL: "",
            }

        if not isinstance(payload, dict):
            return {
                PREFERENCES_KEY_MODELS: [],
                PREFERENCES_KEY_DEFAULT_MODEL: "",
            }

        return self._normalize(payload)

    def save(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.load()
        merged = dict(current)

        if PREFERENCES_KEY_MODELS in updates:
            merged[PREFERENCES_KEY_MODELS] = updates.get(PREFERENCES_KEY_MODELS)
        if PREFERENCES_KEY_DEFAULT_MODEL in updates:
            merged[PREFERENCES_KEY_DEFAULT_MODEL] = updates.get(
                PREFERENCES_KEY_DEFAULT_MODEL
            )

        normalized = self._normalize(merged)
        self.preferences_path.write_text(
            json.dumps(normalized, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            self.preferences_path.chmod(0o600)
        except OSError:
            # Best-effort permissions for cross-platform compatibility.
            pass

        return normalized
