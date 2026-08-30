from __future__ import annotations

"""
Qwen communication client.

Default mode is deterministic template fallback. Live Hugging Face inference
stays disabled until qwen.yaml explicitly allows it AND tools remain off.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "ml" / "config" / "qwen.yaml"


class QwenClientError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_qwen_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["authority"]["financial_decisions"]:
        raise QwenClientError("Qwen cannot make financial decisions")
    if config["authority"]["tools_enabled"] or config["safety"]["qwen_tools_enabled"]:
        raise QwenClientError("Qwen tools must stay disabled")
    if config["communication"]["send_to_customer"]:
        raise QwenClientError("Qwen cannot send customer messages yet")
    if config["safety"]["controlled_execution_authorized"]:
        raise QwenClientError("Qwen config cannot authorize execution")
    return config


def qwen_runtime_status() -> dict[str, Any]:
    config = load_qwen_config()
    return {
        "model_id": config["model"]["model_id"],
        "model_enabled": bool(config["model"]["enabled"]),
        "runtime_mode": config["runtime"]["mode"],
        "allow_live_inference": bool(config["runtime"]["allow_live_inference"]),
        "preview_only": bool(config["communication"]["preview_only"]),
        "send_to_customer": False,
        "tools_enabled": False,
        "financial_authority": False,
    }


def generate_with_qwen(
    *,
    action: str,
    amount_minor: int,
    language: str,
    customer_name: str | None = None,
) -> dict[str, Any]:
    """
    Attempt live Qwen generation. Always fail closed to template mode unless
    the config explicitly enables a local model. Weights are never downloaded
    automatically.
    """
    config = load_qwen_config()
    status = qwen_runtime_status()
    if (
        not config["model"]["enabled"]
        or not config["runtime"]["allow_live_inference"]
        or config["runtime"]["mode"] != "live"
    ):
        return {
            "ok": False,
            "source": "disabled",
            "reason": "qwen_live_inference_disabled",
            "status": status,
        }
    if config["model"]["download_weights"]:
        return {
            "ok": False,
            "source": "disabled",
            "reason": "automatic_weight_download_forbidden",
            "status": status,
        }
    # Live path is intentionally not wired to Transformers yet. Keeping this
    # fail-closed avoids shipping an execution-capable LLM path by accident.
    return {
        "ok": False,
        "source": "unavailable",
        "reason": "qwen_weights_not_loaded",
        "status": status,
        "hint": (
            f"Install/load {config['model']['model_id']} locally later. "
            "Until then RecoverAI uses deterministic templates."
        ),
        "request": {
            "action": action,
            "amount_minor": amount_minor,
            "language": language,
            "customer_name": customer_name,
        },
    }
