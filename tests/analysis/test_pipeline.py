from vocabpruner.analysis.pipeline import run_token_pipeline
from vocabpruner.analysis.scoring import arrays_from_facts, ScoreConfig
from vocabpruner.models import TokenRecord


def token_facts():
    return {
        "columns": {
            "token_id": [1],
            "retain_candidate": [True],
            "protected": [False],
        }
    }


def wordfreq_facts():
    return {
        "columns": {
            "token_id": [1, 1],
            "segment_index": [0, 0],
            "language": ["en", "de"],
            "term": ["hello", "hallo"],
            "mode": ["exact", "exact"],
            "zipf": [3.0, 5.0],
            "rank": [1000, 10],
            "rank_percent": [2.0, 0.01],
        }
    }


def test_arrays_use_max_zipf_for_segment_pass():
    arrays = arrays_from_facts(
        token_facts(),
        wordfreq_facts(),
        ScoreConfig(languages=("en", "de"), wordfreq_min_zipf=4.0),
    )

    assert arrays.best_zipf_by_segment[(1, 0)] == 5.0
    assert arrays.segment_wordfreq_passes[(1, 0)] is True


def test_arrays_use_best_rank_percent_for_segment_pass():
    arrays = arrays_from_facts(
        token_facts(),
        wordfreq_facts(),
        ScoreConfig(languages=("en", "de"), wordfreq_min_zipf=0.0, wordfreq_top_pct=0.1),
    )

    assert arrays.best_rank_percent_by_segment[(1, 0)] == 0.01
    assert arrays.segment_wordfreq_passes[(1, 0)] is True


def test_selected_syntax_literal_survives_cached_pipeline(tmp_path):
    result = run_token_pipeline(
        model="tiny",
        config=ScoreConfig(syntax_lists={"json"}, languages=()),
        records=[TokenRecord(0, "true", "true")],
        cache_root=tmp_path,
        workers=1,
    )

    assert result.retained_ids == [0]
    assert result.scores[0].syntax_lists == ["json"]
