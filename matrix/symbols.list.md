# `tokenpruner/symbols.py` ownership list

## Current functions/classes

- `normalize_presets` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `matching_symbol_presets` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `char_matches_symbol_preset` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `symbol_presets` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `symbol_literals` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `_load_symbol_preset` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `_literal_matches` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `contains_emoji_or_pictograph` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here
- `is_emoji_or_pictograph` [keep] stay in `symbols.py`: symbol preset/matching logic belongs here

## Incoming functions/classes

- `compact_symbol_values` from `tokenpruner/pipeline.py` [move] move here: symbol display normalization belongs with symbols
