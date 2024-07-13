from typing import Any


def prefix_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}
