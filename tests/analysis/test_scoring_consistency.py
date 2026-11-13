import pytest

from vocabpruner.analysis.pipeline import run_token_pipeline
from vocabpruner.analysis.scoring import score_token, ScoreConfig
from vocabpruner.models import TokenRecord


@pytest.mark.parametrize("no_wordfreq", [False, True])
def test_single_token_and_cached_pipeline_share_decisions(tmp_path, no_wordfreq):
    records = [
        TokenRecord(0, "<s>", "<s>", is_special=True),
        TokenRecord(1, "hello", "hello"),
        TokenRecord(2, "Ж", "Ж"),
        TokenRecord(3, "true", "true"),
        TokenRecord(4, "\\u1234", "\\u1234"),
        TokenRecord(5, " ", " "),
        TokenRecord(6, "<0xFF>", "�", is_byte_fallback=True),
    ]
    config = ScoreConfig(languages=(), syntax_lists={"json"}, no_wordfreq=no_wordfreq)
    expected = [score_token(record, config) for record in records]
    fresh = run_token_pipeline(model="tiny", config=config, records=records, cache_root=tmp_path)
    cached = run_token_pipeline(model="tiny", config=config, records=records, cache_root=tmp_path)
    assert not fresh.cache_reused
    assert cached.cache_reused
    assert fresh.scores == cached.scores == expected
