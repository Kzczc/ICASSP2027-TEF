"""Model registry loaded from ``configs/models.yaml``.

Values may reference environment variables as ``${NAME}`` or ``${NAME:-default}``, so no
machine-specific path or API key is stored in the repository.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .data import REPO_ROOT

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "models.yaml"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_model_config(name: str, path: str | os.PathLike | None = None) -> dict[str, Any]:
    import yaml

    config_path = Path(path or DEFAULT_MODEL_CONFIG)
    with config_path.open(encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if name not in registry:
        raise KeyError(f"model {name!r} is not defined in {config_path}; available: {sorted(registry)}")
    config = _expand(registry[name])
    config.setdefault("name", name)
    return config
