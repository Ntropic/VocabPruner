# Python API

The package exports `ScoreConfig`, `TokenRecord`, `score_token`, and
`tokenizer_records` from `vocabpruner`.

## Configuration

`ScoreConfig.from_cli(...)` normalizes list names and validates frequency
thresholds. Arguments: `keep_scripts`, `syntax_lists`, `symbol_lists`,
`languages`, `wordfreq_min_zipf`, `wordfreq_top_pct`, and `dictionaries`
(custom dictionary paths). With neither threshold, frequency filtering is off.
Direct `ScoreConfig()` construction enables filtering at Zipf 4.0 instead;
use `no_wordfreq=True` to disable it explicitly.

```python
from vocabpruner import ScoreConfig, TokenRecord, score_token

config = ScoreConfig.from_cli(
    keep_scripts=["Latin", "Common", "Inherited"],
    syntax_lists=["all"],
    symbol_lists=["all"],
)
record = TokenRecord(token_id=0, raw_token="hello", decoded_text="hello")
result = score_token(record, config)
print(result.status)
print(result.explanations)
```

## Vocabulary records

`TokenRecord(token_id, raw_token, decoded_text, is_special=False,
is_byte_fallback=False)` represents one stored vocabulary entry and its
individual decoding.

`tokenizer_records(tokenizer)` extracts these records from a supported loaded
Hugging Face fast tokenizer, including special-token and byte-token flags.

## Scoring

`score_token(record, config, dictionaries=None)` returns a `TokenScore`.
An explicit dictionary argument maps names to sets of words; otherwise configured
custom dictionary files are loaded.

The result exposes `token`, `status`, `protected`, `script_profile`,
`syntax_lists`, `symbol_lists`, `dictionary_matches`, `wordfreq_matches`, and
`explanations`. Status is `protected`, `retain_candidate`, or
`non_retain_candidate`. See [Scoring system](scoring.md).

Single-token scoring does not finalize the pruning map: pruning additionally
preserves required byte tokens, BPE dependencies, and model configuration IDs.
