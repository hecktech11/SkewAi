"""Small, dependency-free JSON-schema subset validator.  [AI-LAYER]

Model output is untrusted. Even though Meta Model API documents
schema-constrained decoding for ``response_format: json_schema``, every
response is re-validated here before any value reaches product state.

Supported keywords: type (incl. lists and "null"), properties, required,
additionalProperties (bool), items, enum, minimum, maximum, minLength,
maxLength, maxItems, minItems.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _type_ok(value: Any, typ: str) -> bool:
    if typ in ("integer", "number") and isinstance(value, bool):
        return False
    if typ == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _TYPES.get(typ, (object,)))


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of error strings; empty means valid."""
    errors: list[str] = []
    typ = schema.get("type")
    if typ is not None:
        types = typ if isinstance(typ, list) else [typ]
        if not any(_type_ok(value, t) for t in types):
            return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required '{key}'")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected property '{key}'")
        for key, sub in props.items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))
    return errors


__all__ = ["validate"]
