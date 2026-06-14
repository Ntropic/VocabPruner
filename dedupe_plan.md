# Two-Phase Refactor Plan

Checklist derived from `code_map.md` and `matrix/*.matrix.md`. No new modules. No compatibility aliases.

## Prep Verification

- [x] Confirm `code_map.md` has no move/merge owner outside existing `tokenpruner/*.py` files.
- [x] Confirm each `matrix/*.list.md` has current functions plus incoming candidates.
- [x] Confirm each `matrix/*.matrix.md` has existing function columns, `Add function`, and `Delete`.
- [x] Confirm each `G` cell names the target function to merge into.

## Phase 1: Move To Owners

Move functions to their owner files first. Keep behavior as close as possible. Do not dedupe merge rows in this phase except for import/callsite updates needed by the move.

- [x] Move `cached_create_only_pipeline` from `cli.py` to `pipeline.py`.
- [x] Move `write_token_decisions` from `cli.py` to `pipeline.py`.
- [x] Move `write_create_artifacts` from `cli.py` to `pipeline.py`.
- [x] Move `settings_config` and `require_wordfreq_threshold` from `cli.py` to `scoring.py`.
- [x] Move `format_gb` from `cli.py` to `reports.py`.
- [x] Move `ensure_pruning_runtime` from `cli.py` to `pruning.py`.
- [x] Move tokenizer save/rewrite helpers from `pruning.py` to `vocab.py`.
- [x] Move shared tokenizer/config token-id helpers from `pruning.py` to `vocab.py`.
- [x] Move `compact_symbol_values` from `pipeline.py` to `symbols.py`.
- [x] Update imports and callsites for moved functions.
- [x] Run `python -m py_compile tokenpruner/*.py`.
- [x] Run focused tests for moved owners: `uv run pytest tests/test_cli.py tests/test_pipeline.py tests/test_pruning.py tests/test_vocab.py tests/test_scoring_sampling.py tests/test_reports.py tests/test_scripts_symbols.py`.

## Phase 2: Merge And Rename

Handle one merge group at a time. For each group: pick survivor, replace callsites, delete old function, run focused tests, then mark done.

- [x] Merge benchmark `model_slug` into `pipeline.py::model_slug`; remove benchmark duplicate with benchmark deletion.
- [x] Merge CLI artifact path/selection helpers into pipeline output workflow:
  - [x] `resolve_save_dir` into `pipeline.py::write_create_artifacts`.
  - [x] `decision_artifact_dir` into `pipeline.py::write_create_artifacts`.
  - [x] `should_create_list` into `pipeline.py::write_create_artifacts`.
  - [x] `should_create_decisions` into `pipeline.py::write_create_artifacts`.
  - [x] `should_create_html` into `pipeline.py::write_create_artifacts`.
- [x] Merge pruning vocabulary-name policy:
  - [x] `_looks_like_vocabulary_module` into `pruning.py::is_checkpoint_vocab_tensor` through one shared predicate.
  - [x] Rename callsites to shared predicate/survivor.
- [x] Merge tokenizer added-token helpers:
  - [x] `_rewrite_added_tokens_json` into `vocab.py::_remap_added_tokens`.
  - [x] Rename callsites to survivor.
- [x] Merge tokenizer/config token-id helpers:
  - [x] `_remap_json_token_id_fields` into `vocab.py::_remap_config_token_id` or a single shared helper in `vocab.py`.
  - [x] `_remap_token_references` into the same shared JSON token-id recursion helper.
  - [x] Rename callsites to survivor.
- [x] Merge report escaping:
  - [x] Delete markdown-only `_escape` after `render_markdown` removal.
  - [x] Keep `_rich_escape` only where rich plain-text rendering needs it.

## Delete Pass

Delete code already marked `D` after moves/merges cannot depend on it.

- [x] Delete benchmark feature files unless benchmark is explicitly accepted: `tokenpruner/benchmark.py`, `tests/test_benchmark.py`.
- [x] Delete HF cache disk inference fallback; keep marker as source of truth.
- [x] Delete unused markdown report renderer and markdown-only helpers in `reports.py`.
- [x] Delete `validate_text_only_pruning` and `_primary_vocab_size`.
- [x] Delete dead CLI helper `score_config_to_dict`.

## Test Plan

- [x] `python -m py_compile tokenpruner/*.py`
- [x] `uv run pytest tests/test_architecture.py tests/test_dictionaries.py tests/test_pipeline.py tests/test_pruning.py tests/test_reports.py tests/test_scoring_sampling.py tests/test_scripts_symbols.py tests/test_token_lists.py tests/test_vocab.py tests/test_wordfreq_support.py`
- [x] `uv run pytest tests/test_hf_cache.py`
- [x] `uv run pytest tests/test_cli.py`
- [x] `uv run pytest`
