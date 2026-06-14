# `tokenpruner/vocab.py` ownership list

## Current functions/classes

- `is_byte_fallback_token` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here
- `load_tokenizer` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here
- `tokenizer_records` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here
- `decode_token` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here
- `vocabulary_stats` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here
- `_special_token_ids` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here
- `_special_tokens` [keep] stay in `vocab.py`: tokenizer/vocabulary logic belongs here

## Incoming functions/classes

- `save_remapped_tokenizer` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `rewrite_tokenizer_json` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `validate_saved_tokenizer` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `_rewrite_vocab_json` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `_rewrite_tokenizer_config` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `_rewrite_merges_file` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `_merge_is_retained` from `tokenpruner/pruning.py` [move] move here: tokenizer save/rewrite belongs with tokenizer loading and records
- `_rewrite_added_tokens_json` from `tokenpruner/pruning.py` [merge] merge here into `_remap_added_tokens`: added-token rewrite overlaps added-token remap helper; unify in vocab.py
- `_remap_json_token_id_fields` from `tokenpruner/pruning.py` [merge] merge here into `_remap_config_token_id`: JSON token-id remap overlaps config token-id remap; unify helper in vocab.py
- `_remap_token_references` from `tokenpruner/pruning.py` [merge] merge here into `_remap_json_token_id_fields`: token-reference recursion overlaps JSON token-id field recursion; unify in vocab.py
- `_remap_added_tokens` from `tokenpruner/pruning.py` [move] move here: shared tokenizer/config remap helper belongs with tokenizer rewrite
- `_is_token_id_key` from `tokenpruner/pruning.py` [move] move here: shared tokenizer/config remap helper belongs with tokenizer rewrite
- `_remap_config_token_id` from `tokenpruner/pruning.py` [move] move here: shared tokenizer/config remap helper belongs with tokenizer rewrite
