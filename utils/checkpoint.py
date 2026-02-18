from __future__ import annotations

from collections.abc import Mapping


_PREFIXES_TO_STRIP = (
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_orig_mod.",
    "module.",
)

_ROOT_PREFIXES_TO_STRIP = (
    "generator.",
    "generator_ema.",
    "model.generator.",
)


def _is_param_dict(value):
    if not isinstance(value, Mapping) or not value:
        return False
    return all(hasattr(v, "shape") for v in value.values())


def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for name, value in state_dict.items():
        for prefix in _PREFIXES_TO_STRIP:
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _strip_root_prefixes(state_dict):
    """Strip common top-level namespaces used by trainer wrappers."""
    if not state_dict:
        return state_dict

    keys = tuple(state_dict.keys())
    for prefix in _ROOT_PREFIXES_TO_STRIP:
        if all(name.startswith(prefix) for name in keys):
            prefix_len = len(prefix)
            return {name[prefix_len:]: value for name, value in state_dict.items()}

    return state_dict


def _extract_candidate_dicts(checkpoint):
    """Yield state-dict-like mappings inside checkpoint in a stable search order."""
    if _is_param_dict(checkpoint):
        yield "root", checkpoint
        return

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Unsupported checkpoint format: expected mapping-like object.")

    # Highest-priority keys first; recurse through nested wrappers.
    priority_keys = ("generator", "model", "state_dict", "generator_ema", "ema")
    visited = set()

    def _walk(node, path="root"):
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if _is_param_dict(node):
            yield path, node
            return

        if not isinstance(node, Mapping):
            return

        for key in priority_keys:
            value = node.get(key)
            if value is not None:
                yield from _walk(value, f"{path}.{key}")

        for key, value in node.items():
            if key in priority_keys:
                continue
            yield from _walk(value, f"{path}.{key}")

    yield from _walk(checkpoint)


def extract_model_state_dict(checkpoint, prefer_ema=False):
    """
    Extract and normalize a model state_dict from a checkpoint payload.

    Returns:
        tuple[dict, str]: cleaned state_dict and the key/source it came from.
    """
    candidates = list(_extract_candidate_dicts(checkpoint))
    if not candidates:
        available = list(checkpoint.keys())[:20] if isinstance(checkpoint, Mapping) else []
        raise ValueError(
            "Could not find a model state_dict in checkpoint. "
            f"Available keys: {available}"
        )

    # Reorder candidates so EMA namespaces are preferred when requested.
    if prefer_ema:
        candidates.sort(key=lambda item: ("ema" not in item[0], item[0]))
    else:
        candidates.sort(key=lambda item: ("ema" in item[0], item[0]))

    key, state_dict = candidates[0]
    cleaned = _clean_state_dict_keys(state_dict)
    cleaned = _strip_root_prefixes(cleaned)
    return cleaned, key
