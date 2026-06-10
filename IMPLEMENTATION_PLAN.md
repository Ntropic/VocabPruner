# Single Pipeline With Cached Retain-Candidate Lists

## Summary

- [x] Refactor TokenPruner toward one shared pipeline used by `analyze` and `prune`.
- [x] Build or load threshold-independent token facts.
- [x] Enrich retain candidates with wordfreq data.
- [x] Derive current retained/pruned decisions from active threshold settings.
- [x] Remove duplicate CLI paths for old sampled/list-file workflows.

## Key Changes

- [x] Add a central pipeline entrypoint, `run_token_pipeline(...)`.
- [x] Load tokenizer records in command wrappers and pass them into the pipeline.
- [x] Resolve compact cache paths from model plus non-threshold settings.
- [x] Load cached facts when schema/policy match.
- [x] Compute deterministic token facts when cache is missing.
- [x] Enrich retain candidates with Zipf and rank-percent for selected languages.
- [x] Parse cached JSON columns into NumPy arrays.
- [x] Use NumPy arrays for threshold derivation instead of Python dict/list traversal.
- [x] Return one result object with facts, derived decisions, retained ids, pruned ids, cache path, and reuse/create status.
- [x] Make `analyze` and `prune` thin wrappers around the shared pipeline.
- [x] Make `prune` consume retained IDs from the same pipeline result that `analyze` reports.

## Terminology And Decisions

- [x] Keep `retain_candidates` as the term for tokens that pass deterministic script/symbol/protection checks.
- [x] Treat protected/structural tokens as always retained.
- [x] Treat deterministic gate failures as immediately non-retained.
- [x] Retain candidates are retained only if required word segments pass dictionary evidence or current frequency thresholds.
- [x] Zipf mode uses frequency evidence across selected languages.
- [x] Top-percent mode uses rank-percent evidence across selected languages.
- [x] Implement max Zipf and best rank-percent aggregation with NumPy arrays.

## Cache Format And Layout

- [x] Store JSON on disk.
- [x] Use a column-oriented cache shape.
- [x] Add NumPy as an internal dependency.
- [x] Cache path excludes Zipf/top-percent thresholds.
- [x] Cache path includes model, scripts, symbols, dictionaries, and selected languages.
- [x] Use one compact slug function for cache and prune output paths.
- [x] Preserve compact `all` for symbol presets.
- [x] Write `policy.json`.
- [x] Write `token_facts.json`.
- [x] Write `wordfreq_facts.json`.
- [x] Do not write retained/pruned ID files by default.

## CLI

- [x] Replace `--wordfreq-lang` with `--languages`.
- [x] Remove `--sample-size`.
- [x] Remove no-op `--workers`.
- [x] Remove `--retained-token-ids`.
- [x] Remove the `list` subcommand.
- [x] Add `list` as a `--show` mode.
- [x] Remove `--show json`.
- [x] Make `--show list` print JSONL token decisions.
- [x] Keep `--show html` as the HTML viewer output mode.

## HTML

- [x] `--show html` runs the same pipeline.
- [x] `--show html` writes static viewer assets.
- [x] Viewer imports cached JS data.
- [x] Viewer provides a language dropdown.
- [x] Plot renders sampled wordfreq points.
- [x] Hover maps to the sorted full language row list in browser memory.
- [x] Hover shows word, language, Zipf, rank, rank percent, and pass/fail.
- [x] Precompute sorted/compact language arrays from NumPy-backed cache data for faster HTML generation.

## Tests

- [x] Verify cache reuse when only Zipf/top-percent thresholds change.
- [x] Verify threshold changes alter derived decisions without rebuilding facts.
- [x] Verify `analyze` and `prune` use the shared pipeline path.
- [x] Verify `--show list` emits JSONL.
- [x] Verify `--show json` is rejected.
- [x] Verify `--show html` writes viewer assets.
- [x] Verify NumPy-backed sorting/filtering drives retained/pruned derivation.
- [x] Verify NumPy-backed max-Zipf and best-rank-percent aggregation.
- [x] Update README to document the single pipeline and clarify `retain_candidates`.

## Remaining Implementation Work

- [x] Replace `derive_scores()` wordfreq dict/list threshold logic with NumPy-derived segment pass/fail arrays.
- [x] Add focused tests for NumPy threshold derivation.
- [x] Remove old thresholded `wordfreq_matches()` path or make it a wrapper around diagnostics after pipeline scoring no longer needs it.
- [x] Revisit fallback wordfreq data after tests monkeypatch wordfreq behavior directly.
