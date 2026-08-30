from __future__ import annotations

"""Validate structured communication payloads before any preview is shown."""

from typing import Any

from ml.src.qwen_client import load_qwen_config


REQUIRED_FIELDS = ("intent", "language", "message", "confidence")


def validate_communication_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = load_qwen_config()
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        return {
            "valid": False,
            "reason": "malformed_output",
            "missing": missing,
        }
    language = str(payload["language"]).lower()
    if language not in config["communication"]["supported_languages"]:
        return {
            "valid": False,
            "reason": "unsupported_language",
            "language": language,
        }
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError):
        return {"valid": False, "reason": "invalid_confidence"}
    if not 0 <= confidence <= 1:
        return {"valid": False, "reason": "confidence_out_of_range"}
    if config["communication"]["send_to_customer"]:
        return {"valid": False, "reason": "send_to_customer_must_remain_false"}
    return {
        "valid": True,
        "reason": "validated",
        "preview_only": True,
        "send_to_customer": False,
    }
