# TokenPruner Code Map

Step 1 artifact: actual top-level class/function ownership map. No cleanup applied here.

Hard rule: every `{move: ...}` owner and merge owner is an existing `tokenpruner/*.py` file. No new modules, no compatibility aliases.

Decision tags:

- `{keep: current file is natural owner}`
- `{move: target_file.py, reason}`
- `{merge: target_file.py::target_function, reason}`
- `{delete: unused/dead/scope creep}`

Existing owner files allowed: `__init__.py`, `architecture.py`, `benchmark.py`, `cli.py`, `dictionaries.py`, `hf_cache.py`, `models.py`, `paths.py`, `pipeline.py`, `pruning.py`, `reports.py`, `scoring.py`, `scripts.py`, `symbols.py`, `token_lists.py`, `vocab.py`, `wordfreq_support.py`

Placement rules used:

- CLI owns parser, argument normalization, dispatch, prompts, local server launch, and terminal printing.
- Tokenizer loading, records, and tokenizer rewrite helpers belong in `vocab.py`.
- Model/config inspection belongs in `architecture.py`; pruning config rewrite can stay in `pruning.py` until dedupe forces a move.
- HF cache/download completion uses marker logic in `hf_cache.py`; disk inference fallback is delete-candidate.
- Checkpoint source resolution, safetensors validation, and checkpoint pruning helpers stay in `pruning.py` for now.
- HTML viewer remains in `pipeline.py` for now unless extraction is explicitly approved.
- Report rendering stays in `reports.py`; unused markdown rendering is delete-candidate.
- Benchmark code is delete/merge-candidate unless explicitly accepted.

## `tokenpruner/__init__.py`

- No top-level classes/functions. {keep: package export surface belongs here}

## `tokenpruner/architecture.py`

- `ModelArchitectureReport` {keep: architecture inspection/config helper belongs here}
- `load_model_config` {keep: architecture inspection/config helper belongs here}
- `inspect_model_architecture` {keep: architecture inspection/config helper belongs here}
- `render_model_architecture` {keep: architecture inspection/config helper belongs here}
- `_flatten` {keep: architecture inspection/config helper belongs here}
- `_encoder_paths` {keep: architecture inspection/config helper belongs here}
- `_is_vocabulary_size` {keep: architecture inspection/config helper belongs here}
- `_is_mtp_setting` {keep: architecture inspection/config helper belongs here}
- `_is_mtp_path` {keep: architecture inspection/config helper belongs here}
- `_enabled` {keep: architecture inspection/config helper belongs here}

## `tokenpruner/benchmark.py`

- `DatasetSpec` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `TokenDelta` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `SpeedResult` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `LmEvalRun` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `BenchmarkReport` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `load_prompt_file` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `load_dataset_prompts` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `dataset_spec` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `extract_text` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `append_text` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `measure_token_delta` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `token_count` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `run_lm_eval` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `measure_speed` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `write_report` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `report_to_dict` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `render_summary` {delete: benchmark feature is out of cleanup scope unless explicitly accepted}
- `model_slug` {merge: `pipeline.py::model_slug`, duplicate slug helper; use pipeline slug policy, delete benchmark copy}

## `tokenpruner/cli.py`

- `default_workers` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `build_parser` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `add_settings_arguments` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `add_worker_argument` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `add_save_argument` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `add_show_argument` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `add_create_argument` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `inspect_command` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `analyze_command` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `prune_command` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `main` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `app` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `settings_config` {move: `scoring.py`, scoring config policy belongs with ScoreConfig}
- `require_wordfreq_threshold` {move: `scoring.py`, scoring config policy belongs with ScoreConfig}
- `cached_create_only_pipeline` {move: `pipeline.py`, cache/pipeline construction belongs with pipeline}
- `write_token_decisions` {move: `pipeline.py`, list/artifact writing belongs with pipeline outputs until new module approved}
- `write_create_artifacts` {move: `pipeline.py`, list/artifact writing belongs with pipeline outputs until new module approved}
- `start_html_server` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `free_local_port` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `score_config_to_dict` {delete: unused helper}
- `resolve_save_dir` {merge: `pipeline.py::write_create_artifacts`, artifact path/selection helpers overlap output-writing workflow; unify there}
- `decision_artifact_dir` {merge: `pipeline.py::write_create_artifacts`, artifact path/selection helpers overlap output-writing workflow; unify there}
- `format_gb` {move: `reports.py`, display formatting belongs with report/output helpers}
- `print_pruning_preview` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `print_cache_status` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `confirm_pruning` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `ensure_pruning_runtime` {move: `pruning.py`, runtime dependency policy belongs with pruning}
- `print_prune_decisions` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `normalize_show_args` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `normalize_create_args` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}
- `should_create_list` {merge: `pipeline.py::write_create_artifacts`, artifact path/selection helpers overlap output-writing workflow; unify there}
- `should_create_decisions` {merge: `pipeline.py::write_create_artifacts`, artifact path/selection helpers overlap output-writing workflow; unify there}
- `should_create_html` {merge: `pipeline.py::write_create_artifacts`, artifact path/selection helpers overlap output-writing workflow; unify there}
- `parse_examples_spec` {keep: CLI parser, dispatch, prompt, local server, or terminal printing belongs here}

## `tokenpruner/dictionaries.py`

- `load_builtin_dictionaries` {keep: dictionary loading/matching belongs here}
- `load_custom_dictionary` {keep: dictionary loading/matching belongs here}
- `load_dictionaries` {keep: dictionary loading/matching belongs here}
- `dictionary_matches` {keep: dictionary loading/matching belongs here}
- `normalize_token_text` {keep: dictionary loading/matching belongs here}
- `_parse_terms` {keep: dictionary loading/matching belongs here}
- `_match_mode` {keep: dictionary loading/matching belongs here}

## `tokenpruner/hf_cache.py`

- `CachedCheckpointStatus` {keep: HF cache marker/path logic belongs here}
- `repo_cache_dir` {keep: HF cache marker/path logic belongs here}
- `download_marker_path` {keep: HF cache marker/path logic belongs here}
- `read_download_marker` {keep: HF cache marker/path logic belongs here}
- `mark_analysis_download` {keep: HF cache marker/path logic belongs here}
- `mark_checkpoint_download` {keep: HF cache marker/path logic belongs here}
- `mark_download` {keep: HF cache marker/path logic belongs here}
- `cached_checkpoint_status` {keep: HF cache marker/path logic belongs here}
- `infer_cached_checkpoint_status` {delete: disk inference fallback can accept partial downloads; completion marker is source of truth}
- `snapshot_has_complete_safetensors` {delete: disk inference fallback can accept partial downloads; completion marker is source of truth}
- `snapshot_has_checkpoint_reference` {delete: disk inference fallback can accept partial downloads; completion marker is source of truth}
- `snapshot_has_analysis_files` {delete: disk inference fallback can accept partial downloads; completion marker is source of truth}

## `tokenpruner/models.py`

- `TokenRecord` {keep: shared data model belongs here}
- `ScriptProfile` {keep: shared data model belongs here}
- `DictionaryMatch` {keep: shared data model belongs here}
- `WordfreqMatch` {keep: shared data model belongs here}
- `TokenScore` {keep: shared data model belongs here}

## `tokenpruner/paths.py`

- No top-level classes/functions. {keep: package path constants belong here}

## `tokenpruner/pipeline.py`

- `PipelineResult` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `PipelineArrays` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `write_wordfreq_html` {keep: HTML viewer stays in existing pipeline.py until extraction explicitly approved}
- `_viewer_relative_path` {keep: HTML viewer stays in existing pipeline.py until extraction explicitly approved}
- `_wordfreq_html` {keep: HTML viewer stays in existing pipeline.py until extraction explicitly approved}
- `run_token_pipeline` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `token_cache_dir` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `compact_settings_slug` {keep: slug helper chain already shares local policy; no separate artifact module approved}
- `list_settings` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `load_cached_facts` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `write_cached_facts` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `build_token_facts` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_token_fact_rows` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_token_fact_row_from_args` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_token_fact_row` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `deterministic_token_fact` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `arrays_from_facts` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_best_by_segment` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_segment_passes` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `derive_scores` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_diagnostic_to_match` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `_diagnostic_passes_threshold` {keep: pipeline orchestration/cache/scoring prep belongs here}
- `compact_symbol_values` {move: `symbols.py`, symbol display normalization belongs with symbols}
- `symbol_slug` {keep: slug helper chain already shares local policy; no separate artifact module approved}
- `threshold_slug` {keep: slug helper chain already shares local policy; no separate artifact module approved}
- `value_slug` {keep: slug helper chain already shares local policy; no separate artifact module approved}
- `number_slug` {keep: slug helper chain already shares local policy; no separate artifact module approved}
- `model_slug` {keep: single accepted model slug owner; benchmark duplicate merges here}
- `_progress` {keep: pipeline orchestration/cache/scoring prep belongs here}

## `tokenpruner/pruning.py`

- `PruneRefusal` {keep: pruning transform/runtime helper belongs here}
- `PruneResult` {keep: pruning transform/runtime helper belongs here}
- `TransformedSurface` {keep: pruning transform/runtime helper belongs here}
- `validate_pruning_architecture` {keep: pruning transform/runtime helper belongs here}
- `model_vocabulary_size` {keep: pruning transform/runtime helper belongs here}
- `validate_pruning_runtime` {keep: pruning transform/runtime helper belongs here}
- `missing_pruning_packages` {keep: pruning transform/runtime helper belongs here}
- `_module_available` {keep: pruning transform/runtime helper belongs here}
- `validate_text_only_pruning` {delete: unused text-only compatibility path}
- `_primary_vocab_size` {delete: unused text-only compatibility path}
- `load_causal_lm` {keep: full model pruning path stays in pruning.py; no new module allowed}
- `prune_model_and_tokenizer` {keep: full model pruning path stays in pruning.py; no new module allowed}
- `prune_safetensors_checkpoint` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `resolve_checkpoint_source` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `copy_checkpoint_sidecars` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `write_pruned_config` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `load_safetensors_index` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `checkpoint_safetensor_paths` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `validate_safetensor_files` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `write_safetensors_index` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `is_checkpoint_vocab_tensor` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `record_checkpoint_surface` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `checkpoint_module_name` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `checkpoint_file_size` {keep: checkpoint pruning helpers stay in pruning.py for now; no new module allowed}
- `slice_model_vocab` {keep: pruning transform/runtime helper belongs here}
- `_model_vocab_size` {keep: pruning transform/runtime helper belongs here}
- `discover_vocabulary_surfaces` {keep: pruning transform/runtime helper belongs here}
- `_looks_like_vocabulary_module` {merge: `pruning.py::is_checkpoint_vocab_tensor`, same vocabulary-name marker policy; one shared predicate in pruning.py}
- `_surface_kind` {keep: pruning transform/runtime helper belongs here}
- `_module_has_vocab_rows` {keep: pruning transform/runtime helper belongs here}
- `_module_row_tensors` {keep: pruning transform/runtime helper belongs here}
- `_set_direct_tensor` {keep: pruning transform/runtime helper belongs here}
- `_update_module_vocab_metadata` {keep: pruning transform/runtime helper belongs here}
- `save_remapped_tokenizer` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `rewrite_tokenizer_json` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `validate_saved_tokenizer` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `_rewrite_vocab_json` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `_rewrite_added_tokens_json` {merge: `vocab.py::_remap_added_tokens`, added-token rewrite overlaps added-token remap helper; unify in vocab.py}
- `_rewrite_tokenizer_config` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `_remap_json_token_id_fields` {merge: `vocab.py::_remap_config_token_id`, JSON token-id remap overlaps config token-id remap; unify helper in vocab.py}
- `_rewrite_merges_file` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `_merge_is_retained` {move: `vocab.py`, tokenizer save/rewrite belongs with tokenizer loading and records}
- `write_pruning_maps` {keep: pruning-specific artifact/config rewrite stays in pruning.py; no new config module allowed}
- `remap_model_config` {keep: pruning-specific artifact/config rewrite stays in pruning.py; no new config module allowed}
- `_remap_config_value` {keep: pruning-specific artifact/config rewrite stays in pruning.py; no new config module allowed}
- `_is_token_id_key` {move: `vocab.py`, shared tokenizer/config remap helper belongs with tokenizer rewrite}
- `_remap_config_token_id` {move: `vocab.py`, shared tokenizer/config remap helper belongs with tokenizer rewrite}
- `_remapped_padding_idx` {keep: pruning transform/runtime helper belongs here}
- `_remap_added_tokens` {move: `vocab.py`, shared tokenizer/config remap helper belongs with tokenizer rewrite}
- `_remap_token_references` {merge: `vocab.py::_remap_json_token_id_fields`, token-reference recursion overlaps JSON token-id field recursion; unify in vocab.py}

## `tokenpruner/reports.py`

- `ExampleCounts` {keep: report rendering/filtering helper belongs here}
- `render_jsonl` {keep: report rendering/filtering helper belongs here}
- `render_markdown` {delete: unused markdown renderer}
- `render_plain_text` {keep: report rendering/filtering helper belongs here}
- `filter_scores` {keep: report rendering/filtering helper belongs here}
- `should_show_rows` {keep: report rendering/filtering helper belongs here}
- `should_show_examples` {keep: report rendering/filtering helper belongs here}
- `normalize_show_options` {keep: report rendering/filtering helper belongs here}
- `score_stats` {keep: report rendering/filtering helper belongs here}
- `example_groups` {keep: report rendering/filtering helper belongs here}
- `_seeded_examples` {keep: report rendering/filtering helper belongs here}
- `_table` {delete: helper only supports deleted markdown renderer}
- `_plain_rows` {keep: report rendering/filtering helper belongs here}
- `_escape` {merge: `reports.py::_rich_escape`, escaping helpers overlap; keep only renderer-needed escaping after markdown deletion}
- `_rich_escape` {keep: report rendering/filtering helper belongs here}
- `_show_label` {keep: report rendering/filtering helper belongs here}

## `tokenpruner/scoring.py`

- `ScoreConfig` {keep: scoring/config/token segmentation belongs here}
- `score_token` {keep: scoring/config/token segmentation belongs here}
- `decision_tree_status` {keep: scoring/config/token segmentation belongs here}
- `_decision_explanation` {keep: scoring/config/token segmentation belongs here}
- `CharacterGate` {keep: scoring/config/token segmentation belongs here}
- `character_gate` {keep: scoring/config/token segmentation belongs here}
- `is_structural_token` {keep: scoring/config/token segmentation belongs here}
- `is_control_char` {keep: scoring/config/token segmentation belongs here}
- `TokenSegment` {keep: scoring/config/token segmentation belongs here}
- `token_segments` {keep: scoring/config/token segmentation belongs here}
- `language_word_segments` {keep: scoring/config/token segmentation belongs here}
- `language_evidence_segments` {keep: scoring/config/token segmentation belongs here}
- `split_identifier_words` {keep: scoring/config/token segmentation belongs here}
- `greedy_dictionary_split` {keep: scoring/config/token segmentation belongs here}
- `_dictionary_split_terms` {keep: scoring/config/token segmentation belongs here}
- `_greedy_split` {keep: scoring/config/token segmentation belongs here}
- `is_word_char` {keep: scoring/config/token segmentation belongs here}
- `has_language_evidence` {keep: scoring/config/token segmentation belongs here}

## `tokenpruner/scripts.py`

- `script_for_char` {keep: Unicode script logic belongs here}
- `script_profile` {keep: Unicode script logic belongs here}
- `normalize_scripts` {keep: Unicode script logic belongs here}

## `tokenpruner/symbols.py`

- `normalize_presets` {keep: symbol preset/matching logic belongs here}
- `matching_symbol_presets` {keep: symbol preset/matching logic belongs here}
- `char_matches_symbol_preset` {keep: symbol preset/matching logic belongs here}
- `symbol_presets` {keep: symbol preset/matching logic belongs here}
- `symbol_literals` {keep: symbol preset/matching logic belongs here}
- `_load_symbol_preset` {keep: symbol preset/matching logic belongs here}
- `_literal_matches` {keep: symbol preset/matching logic belongs here}
- `contains_emoji_or_pictograph` {keep: symbol preset/matching logic belongs here}
- `is_emoji_or_pictograph` {keep: symbol preset/matching logic belongs here}

## `tokenpruner/token_lists.py`

- `compact_token_maps` {keep: token ID map utility belongs here}

## `tokenpruner/vocab.py`

- `is_byte_fallback_token` {keep: tokenizer/vocabulary logic belongs here}
- `load_tokenizer` {keep: tokenizer/vocabulary logic belongs here}
- `tokenizer_records` {keep: tokenizer/vocabulary logic belongs here}
- `decode_token` {keep: tokenizer/vocabulary logic belongs here}
- `vocabulary_stats` {keep: tokenizer/vocabulary logic belongs here}
- `_special_token_ids` {keep: tokenizer/vocabulary logic belongs here}
- `_special_tokens` {keep: tokenizer/vocabulary logic belongs here}

## `tokenpruner/wordfreq_support.py`

- `WordfreqDiagnostic` {keep: wordfreq matching/backend helper belongs here}
- `wordfreq_matches` {keep: wordfreq matching/backend helper belongs here}
- `wordfreq_diagnostics` {keep: wordfreq matching/backend helper belongs here}
- `_match` {keep: wordfreq matching/backend helper belongs here}
- `_wordfreq_match_mode` {keep: wordfreq matching/backend helper belongs here}
- `_passes_wordfreq_threshold` {keep: wordfreq matching/backend helper belongs here}
- `_word_in_top_pct` {keep: wordfreq matching/backend helper belongs here}
- `_wordlist` {keep: wordfreq matching/backend helper belongs here}
- `_zipf_frequency` {keep: wordfreq matching/backend helper belongs here}
- `_word_rank_index` {keep: wordfreq matching/backend helper belongs here}
- `_rank_info` {keep: wordfreq matching/backend helper belongs here}

## Notes For Matrix Files

- One matrix file per existing package file.
- Rows = functions/classes currently in that file plus incoming move/merge candidates from this map.
- Columns = existing functions/classes in that file, plus `Add function`, plus `Delete`.
- Cell values: `K` keep here, `M` move/add here, `G` merge into that function, `D` delete, `-` not applicable.
- Final column = exact decision copied from this map.
