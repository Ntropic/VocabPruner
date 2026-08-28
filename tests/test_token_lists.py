from vocabpruner.token_lists import compact_token_maps


def test_compact_token_maps_preserve_old_id_order():
    old_to_new, new_to_old = compact_token_maps([8, 2, 5])

    assert old_to_new == {2: 0, 5: 1, 8: 2}
    assert new_to_old == {0: 2, 1: 5, 2: 8}
