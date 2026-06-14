# `tokenpruner/vocab.py` function merge matrix

| Function | Source | `is_byte_fallback_token` | `load_tokenizer` | `tokenizer_records` | `decode_token` | `vocabulary_stats` | `_special_token_ids` | `_special_tokens` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `is_byte_fallback_token` | `tokenpruner/vocab.py` | K | - | - | - | - | - | - | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `load_tokenizer` | `tokenpruner/vocab.py` | - | K | - | - | - | - | - | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `tokenizer_records` | `tokenpruner/vocab.py` | - | - | K | - | - | - | - | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `decode_token` | `tokenpruner/vocab.py` | - | - | - | K | - | - | - | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `vocabulary_stats` | `tokenpruner/vocab.py` | - | - | - | - | K | - | - | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `_special_token_ids` | `tokenpruner/vocab.py` | - | - | - | - | - | K | - | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `_special_tokens` | `tokenpruner/vocab.py` | - | - | - | - | - | - | K | - | - | keep in `vocab.py`: tokenizer/vocabulary logic belongs here |
| `save_remapped_tokenizer` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `rewrite_tokenizer_json` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `validate_saved_tokenizer` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `_rewrite_vocab_json` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `_rewrite_tokenizer_config` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `_rewrite_merges_file` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `_merge_is_retained` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: tokenizer save/rewrite belongs with tokenizer loading and records |
| `_rewrite_added_tokens_json` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | G -> `_remap_added_tokens` | - | merge into `vocab.py::_remap_added_tokens`: added-token rewrite overlaps added-token remap helper; unify in vocab.py |
| `_remap_json_token_id_fields` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | G -> `_remap_config_token_id` | - | merge into `vocab.py::_remap_config_token_id`: JSON token-id remap overlaps config token-id remap; unify helper in vocab.py |
| `_remap_token_references` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | G -> `_remap_json_token_id_fields` | - | merge into `vocab.py::_remap_json_token_id_fields`: token-reference recursion overlaps JSON token-id field recursion; unify in vocab.py |
| `_remap_added_tokens` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: shared tokenizer/config remap helper belongs with tokenizer rewrite |
| `_is_token_id_key` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: shared tokenizer/config remap helper belongs with tokenizer rewrite |
| `_remap_config_token_id` | `tokenpruner/pruning.py` | - | - | - | - | - | - | - | M | - | move to `vocab.py`: shared tokenizer/config remap helper belongs with tokenizer rewrite |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
