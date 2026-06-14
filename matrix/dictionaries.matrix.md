# `tokenpruner/dictionaries.py` function merge matrix

| Function | Source | `load_builtin_dictionaries` | `load_custom_dictionary` | `load_dictionaries` | `dictionary_matches` | `normalize_token_text` | `_parse_terms` | `_match_mode` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `load_builtin_dictionaries` | `tokenpruner/dictionaries.py` | K | - | - | - | - | - | - | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |
| `load_custom_dictionary` | `tokenpruner/dictionaries.py` | - | K | - | - | - | - | - | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |
| `load_dictionaries` | `tokenpruner/dictionaries.py` | - | - | K | - | - | - | - | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |
| `dictionary_matches` | `tokenpruner/dictionaries.py` | - | - | - | K | - | - | - | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |
| `normalize_token_text` | `tokenpruner/dictionaries.py` | - | - | - | - | K | - | - | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |
| `_parse_terms` | `tokenpruner/dictionaries.py` | - | - | - | - | - | K | - | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |
| `_match_mode` | `tokenpruner/dictionaries.py` | - | - | - | - | - | - | K | - | - | keep in `dictionaries.py`: dictionary loading/matching belongs here |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
