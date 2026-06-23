"""JSON Schema validation utilities.

Validation is best-effort: errors are logged but never block the request.
The resolver is intentionally simple — schemas are loaded from the local
directory and $ref is resolved by filename convention.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_DIR = Path(__file__).parent
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def _load(name: str, version: int) -> dict[str, Any]:
    key = f"{name}.v{version}"
    if key not in _SCHEMA_CACHE:
        path = _SCHEMA_DIR / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        with open(path) as f:
            _SCHEMA_CACHE[key] = json.load(f)
    return _SCHEMA_CACHE[key]


def load_schema(schema_name: str, version: int) -> dict[str, Any]:
    return _load(schema_name, version)


def validate_against_schema(data: dict[str, Any], schema_name: str, version: int) -> None:
    """Validate *data* against a local JSON schema.

    Raises ``jsonschema.ValidationError`` only if the schema itself is found
    and the data fails validation.  Missing schemas are silently ignored.
    """
    try:
        schema = _load(schema_name, version)
    except FileNotFoundError:
        return

    store: dict[str, dict[str, Any]] = {}

    # Pre-load known local schemas so $ref works without network access.
    for path in _SCHEMA_DIR.glob("*.json"):
        try:
            with open(path) as f:
                s = json.load(f)
            # Register by relative path references (e.g. "./agent_config.v1.json")
            store[f"./{path.name}"] = s
            store[path.name] = s
            if "$id" in s:
                store[s["$id"]] = s
        except Exception:
            pass

    resolver = jsonschema.RefResolver(
        base_uri=f"file://{_SCHEMA_DIR.absolute()}/",
        referrer=schema,
        store=store,
    )

    validator = jsonschema.Draft202012Validator(schema, resolver=resolver)
    errors = list(validator.iter_errors(data))
    if errors:
        messages = []
        for err in errors:
            path_str = " -> ".join(str(p) for p in err.path)
            messages.append(f"{path_str}: {err.message}" if err.path else err.message)
        raise jsonschema.ValidationError(
            f"Validation failed for '{schema_name}' v{version}:\n"
            + "\n".join(f"  - {m}" for m in messages)
        )
