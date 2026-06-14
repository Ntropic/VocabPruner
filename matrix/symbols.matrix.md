# `tokenpruner/symbols.py` function merge matrix

| Function | Source | `normalize_presets` | `matching_symbol_presets` | `char_matches_symbol_preset` | `symbol_presets` | `symbol_literals` | `_load_symbol_preset` | `_literal_matches` | `contains_emoji_or_pictograph` | `is_emoji_or_pictograph` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `normalize_presets` | `tokenpruner/symbols.py` | K | - | - | - | - | - | - | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `matching_symbol_presets` | `tokenpruner/symbols.py` | - | K | - | - | - | - | - | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `char_matches_symbol_preset` | `tokenpruner/symbols.py` | - | - | K | - | - | - | - | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `symbol_presets` | `tokenpruner/symbols.py` | - | - | - | K | - | - | - | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `symbol_literals` | `tokenpruner/symbols.py` | - | - | - | - | K | - | - | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `_load_symbol_preset` | `tokenpruner/symbols.py` | - | - | - | - | - | K | - | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `_literal_matches` | `tokenpruner/symbols.py` | - | - | - | - | - | - | K | - | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `contains_emoji_or_pictograph` | `tokenpruner/symbols.py` | - | - | - | - | - | - | - | K | - | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `is_emoji_or_pictograph` | `tokenpruner/symbols.py` | - | - | - | - | - | - | - | - | K | - | - | keep in `symbols.py`: symbol preset/matching logic belongs here |
| `compact_symbol_values` | `tokenpruner/pipeline.py` | - | - | - | - | - | - | - | - | - | M | - | move to `symbols.py`: symbol display normalization belongs with symbols |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
