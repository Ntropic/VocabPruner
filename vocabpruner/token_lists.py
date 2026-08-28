from __future__ import annotations


def compact_token_maps(retained_token_ids: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    ordered = sorted(int(token_id) for token_id in retained_token_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("retained token IDs must be unique")
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(ordered)}
    new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
    return old_to_new, new_to_old
