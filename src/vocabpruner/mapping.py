from __future__ import annotations

from typing import Any

from vocabpruner.errors import VocabRewriteError


def compact_token_maps(retained_token_ids: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    ordered = sorted(int(token_id) for token_id in retained_token_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("retained token IDs must be unique")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(ordered)}
    new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
    return old_to_new, new_to_old


def remap_json_token_ids(
    value: Any,
    old_to_new: dict[int, int],
    *,
    path: str,
    skip_keys: set[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in skip_keys:
                continue
            child_path = f"{path}.{key_text}"
            if is_token_id_key(key_text) or (key_text == "ids" and ".special_tokens" in path):
                value[key] = remap_config_token_id(child, old_to_new, child_path)
            elif key_text == "id" and isinstance(child, int) and child in old_to_new:
                value[key] = old_to_new[child]
            else:
                remap_json_token_ids(child, old_to_new, path=child_path, skip_keys=skip_keys)
    elif isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int) and value[1] in old_to_new:
            value[1] = old_to_new[value[1]]
            return
        for index, child in enumerate(value):
            remap_json_token_ids(child, old_to_new, path=f"{path}[{index}]", skip_keys=skip_keys)


def is_token_id_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith("_token_id") or lowered.endswith("_token_ids")


def remap_config_token_id(value: Any, old_to_new: dict[int, int], path: str) -> Any:
    if value is None:
        return None
    if isinstance(value, int):
        if value not in old_to_new:
            raise VocabRewriteError(f"Required configuration token {path}={value} was not retained.")
        return old_to_new[value]
    if isinstance(value, (list, tuple)):
        remapped = [remap_config_token_id(item, old_to_new, path) for item in value]
        return type(value)(remapped)
    return value


def remap_added_tokens(value: Any, old_to_new: dict[int, int]) -> list[Any] | dict[str, int]:
    if isinstance(value, dict):
        return {
            str(token): old_to_new[int(token_id)]
            for token, token_id in value.items()
            if isinstance(token_id, int) and token_id in old_to_new
        }
    if not isinstance(value, list):
        return []
    remapped = []
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), int):
            continue
        old_id = int(entry["id"])
        if old_id not in old_to_new:
            continue
        updated = dict(entry)
        updated["id"] = old_to_new[old_id]
        remapped.append(updated)
    return remapped
