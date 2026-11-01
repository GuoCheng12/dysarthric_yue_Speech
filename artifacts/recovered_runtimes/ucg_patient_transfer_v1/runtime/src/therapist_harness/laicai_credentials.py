"""Use the configured gateway credential without copying it into experiment files."""

import json
import os
from pathlib import Path
import tomllib


def load_laicai_key():
    if os.environ.get("OPENAI_API_KEY"):
        return {"credential_source": "OPENAI_API_KEY environment"}
    backup = Path("/data/qwen3-asr/codex/backups/pre-device-login-20260824T173019Z")
    config = tomllib.loads((backup / "config.toml").read_text())
    provider = config["model_providers"][config["model_provider"]]
    if provider["base_url"].rstrip("/") != "https://api.laicaishare.xyz":
        raise RuntimeError("stored credential is not bound to the requested laicai provider")
    key = json.loads((backup / "auth.json").read_text()).get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("configure OPENAI_API_KEY for the laicai gateway")
    os.environ["OPENAI_API_KEY"] = key
    return {"credential_source": "existing laicai configuration backup; loaded into process memory"}
