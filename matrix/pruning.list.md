# `tokenpruner/pruning.py` ownership list

## Current functions/classes

- `PruneRefusal` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `PruneResult` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `TransformedSurface` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `validate_pruning_architecture` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `model_vocabulary_size` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `validate_pruning_runtime` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `missing_pruning_packages` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_module_available` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `validate_text_only_pruning` [delete] delete: unused text-only compatibility path
- `_primary_vocab_size` [delete] delete: unused text-only compatibility path
- `load_causal_lm` [keep] stay in `pruning.py`: full model pruning path stays in pruning.py; no new module allowed
- `prune_model_and_tokenizer` [keep] stay in `pruning.py`: full model pruning path stays in pruning.py; no new module allowed
- `prune_safetensors_checkpoint` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `resolve_checkpoint_source` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `copy_checkpoint_sidecars` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `write_pruned_config` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `load_safetensors_index` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `checkpoint_safetensor_paths` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `validate_safetensor_files` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `write_safetensors_index` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `is_checkpoint_vocab_tensor` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `record_checkpoint_surface` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `checkpoint_module_name` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `checkpoint_file_size` [keep] stay in `pruning.py`: checkpoint pruning helpers stay in pruning.py for now; no new module allowed
- `slice_model_vocab` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_model_vocab_size` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `discover_vocabulary_surfaces` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_looks_like_vocabulary_module` [merge] merge into `pruning.py::is_checkpoint_vocab_tensor`: same vocabulary-name marker policy; one shared predicate in pruning.py
- `_surface_kind` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_module_has_vocab_rows` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_module_row_tensors` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_set_direct_tensor` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_update_module_vocab_metadata` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `save_remapped_tokenizer` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `rewrite_tokenizer_json` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `validate_saved_tokenizer` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `_rewrite_vocab_json` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `_rewrite_added_tokens_json` [merge] merge into `vocab.py::_remap_added_tokens`: added-token rewrite overlaps added-token remap helper; unify in vocab.py
- `_rewrite_tokenizer_config` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `_remap_json_token_id_fields` [merge] merge into `vocab.py::_remap_config_token_id`: JSON token-id remap overlaps config token-id remap; unify helper in vocab.py
- `_rewrite_merges_file` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `_merge_is_retained` [move] move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records
- `write_pruning_maps` [keep] stay in `pruning.py`: pruning-specific artifact/config rewrite stays in pruning.py; no new config module allowed
- `remap_model_config` [keep] stay in `pruning.py`: pruning-specific artifact/config rewrite stays in pruning.py; no new config module allowed
- `_remap_config_value` [keep] stay in `pruning.py`: pruning-specific artifact/config rewrite stays in pruning.py; no new config module allowed
- `_is_token_id_key` [move] move to `vocab.py`: shared tokenizer/config remap helper belongs with tokenizer rewrite
- `_remap_config_token_id` [move] move to `vocab.py`: shared tokenizer/config remap helper belongs with tokenizer rewrite
- `_remapped_padding_idx` [keep] stay in `pruning.py`: pruning transform/runtime helper belongs here
- `_remap_added_tokens` [move] move to `vocab.py`: shared tokenizer/config remap helper belongs with tokenizer rewrite
- `_remap_token_references` [merge] merge into `vocab.py::_remap_json_token_id_fields`: token-reference recursion overlaps JSON token-id field recursion; unify in vocab.py

## Incoming functions/classes

- `ensure_pruning_runtime` from `tokenpruner/cli.py` [move] move here: runtime dependency policy belongs with pruning
