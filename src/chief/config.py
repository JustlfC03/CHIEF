from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _deep_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = mapping
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping")
        node = child
    node[keys[-1]] = value


def parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ConfigError(f"Override must be KEY=VALUE, got {value!r}")
    key, raw = value.split("=", 1)
    if not key:
        raise ConfigError("Override key cannot be empty")
    return key, yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ConfigError("Top-level YAML value must be a mapping")
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, value = parse_override(item)
        _deep_set(cfg, key, value)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["config_path"] = str(path.resolve())
    return cfg


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        node = node[key]
    return node
