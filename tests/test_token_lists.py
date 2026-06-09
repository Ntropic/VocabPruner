import json

from tokenpruner.models import ScriptProfile, TokenRecord, TokenScore
from tokenpruner.scoring import ScoreConfig
from tokenpruner.token_lists import compact_token_maps, split_token_ids, write_token_list_artifacts


def make_score(token_id: int, status: str) -> TokenScore:
    return TokenScore(
        token=TokenRecord(token_id, f"raw{token_id}", f"text{token_id}"),
        script_profile=ScriptProfile({"Latin": 1}, total_meaningful=1, low_confidence=False),
        protected=status == "protected",
        status=status,
    )


def test_split_token_ids_uses_decision_status():
    scores = [
        make_score(4, "non_retain_candidate"),
        make_score(2, "protected"),
        make_score(7, "retain_candidate"),
    ]

    retained, pruned = split_token_ids(scores)

    assert retained == [2, 7]
    assert pruned == [4]


def test_compact_token_maps_preserve_old_id_order():
    old_to_new, new_to_old = compact_token_maps([8, 2, 5])

    assert old_to_new == {2: 0, 5: 1, 8: 2}
    assert new_to_old == {0: 2, 1: 5, 2: 8}


def test_write_token_list_artifacts(tmp_path):
    scores = [
        make_score(0, "protected"),
        make_score(1, "retain_candidate"),
        make_score(2, "non_retain_candidate"),
    ]

    paths = write_token_list_artifacts(tmp_path, model="tiny", config=ScoreConfig(wordfreq_languages=()), scores=scores)

    assert paths["retained"].read_text(encoding="utf-8") == "0\n1\n"
    assert paths["pruned"].read_text(encoding="utf-8") == "2\n"
    assert len(paths["decisions"].read_text(encoding="utf-8").strip().splitlines()) == 3
    policy = json.loads(paths["policy"].read_text(encoding="utf-8"))
    assert policy["model"] == "tiny"
    assert policy["counts"]["retained_ids"] == 2
    assert policy["counts"]["pruned_ids"] == 1
