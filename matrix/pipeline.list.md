# `tokenpruner/pipeline.py` ownership list

## Current functions/classes

- `PipelineResult` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `PipelineArrays` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `write_wordfreq_html` [keep] stay in `pipeline.py`: HTML viewer stays in existing pipeline.py until extraction explicitly approved
- `_viewer_relative_path` [keep] stay in `pipeline.py`: HTML viewer stays in existing pipeline.py until extraction explicitly approved
- `_wordfreq_html` [keep] stay in `pipeline.py`: HTML viewer stays in existing pipeline.py until extraction explicitly approved
- `run_token_pipeline` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `token_cache_dir` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `compact_settings_slug` [keep] stay in `pipeline.py`: slug helper chain already shares local policy; no separate artifact module approved
- `list_settings` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `load_cached_facts` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `write_cached_facts` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `build_token_facts` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_token_fact_rows` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_token_fact_row_from_args` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_token_fact_row` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `deterministic_token_fact` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `arrays_from_facts` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_best_by_segment` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_segment_passes` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `derive_scores` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_diagnostic_to_match` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `_diagnostic_passes_threshold` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here
- `compact_symbol_values` [move] move to `symbols.py`: symbol display normalization belongs with symbols
- `symbol_slug` [keep] stay in `pipeline.py`: slug helper chain already shares local policy; no separate artifact module approved
- `threshold_slug` [keep] stay in `pipeline.py`: slug helper chain already shares local policy; no separate artifact module approved
- `value_slug` [keep] stay in `pipeline.py`: slug helper chain already shares local policy; no separate artifact module approved
- `number_slug` [keep] stay in `pipeline.py`: slug helper chain already shares local policy; no separate artifact module approved
- `model_slug` [keep] stay in `pipeline.py`: single accepted model slug owner; benchmark duplicate merges here
- `_progress` [keep] stay in `pipeline.py`: pipeline orchestration/cache/scoring prep belongs here

## Incoming functions/classes

- `model_slug` from `tokenpruner/benchmark.py` [merge] merge here into `model_slug`: duplicate slug helper; use pipeline slug policy, delete benchmark copy
- `cached_create_only_pipeline` from `tokenpruner/cli.py` [move] move here: cache/pipeline construction belongs with pipeline
- `write_token_decisions` from `tokenpruner/cli.py` [move] move here: list/artifact writing belongs with pipeline outputs until new module approved
- `write_create_artifacts` from `tokenpruner/cli.py` [move] move here: list/artifact writing belongs with pipeline outputs until new module approved
- `resolve_save_dir` from `tokenpruner/cli.py` [merge] merge here into `write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `decision_artifact_dir` from `tokenpruner/cli.py` [merge] merge here into `write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `should_create_list` from `tokenpruner/cli.py` [merge] merge here into `write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `should_create_decisions` from `tokenpruner/cli.py` [merge] merge here into `write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `should_create_html` from `tokenpruner/cli.py` [merge] merge here into `write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
