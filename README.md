# TokenPruner

TokenPruner analyzes tokenizer vocabularies and decides which tokens to keep for a
target workload. It combines character scripts, symbol presets, dictionaries, and
`wordfreq` language evidence with a deterministic accept/prune decision tree.

## Commands

- `inspect`: load a tokenizer/model config and report vocabulary metadata.
  - `--model`: Hugging Face model ID or local path.
  - `--trust-remote-code`: pass through to Hugging Face loaders.
- `analyze`: classify sampled or full vocabularies and print/save diagnostics.
  - `--model`
  - policy parameters: `--keep-script`, `--drop-script`, `--symbol-preset`,
    `--dictionary`, `--wordfreq-lang`, `--wordfreq-min-zipf`, `--wordfreq-top-pct`
  - sampling/display: `--sample-size`, `--seed`, `--workers`, `--show`, `--save`
- `list`: classify the full vocabulary and write reusable token ID artifacts.
  - `--model`, `--output`
  - policy parameters
  - `--workers`
- `prune`: prune a supported model with compact text-token IDs.
  - `--model`, `--output`
  - policy parameters, or `--retained-token-ids` from `tokenpruner list`
  - `--workers`, `--show`, `--yes`

`list` writes `retained_token_ids.txt`, `pruned_token_ids.txt`,
`token_decisions.jsonl`, and `policy.json`. `prune` writes the remapped model plus
`old_to_new_token_ids.json`, `new_to_old_token_ids.json`, and
`pruning_report.json`.

## Scripts

`--keep-script` and `--drop-script` accept these script names:

- `Latin`, `Greek`, `Cyrillic`
- `Arabic`, `Hebrew`, `Devanagari`, `Thai`
- `Han`, `Hiragana`, `Katakana`, `Hangul`
- `Common` for punctuation, symbols, numbers, separators, and controls
- `Inherited` for combining marks

Script selection controls which character families may pass the character gate. It
does not select a language dictionary. For example, `Latin` can contain English,
French, German, and many other languages.

## Symbols

`--symbol-preset` accepts these preset names:

- `coding`, `shell`, `json`, `yaml`, `xml`, `markdown`
- `mathematics` for operators and common Greek math/science letters such as `π`, `λ`, and `Ω`
- `ascii-punctuation` for ASCII punctuation characters
- `emoji` for emoji and pictographic symbols
- `all` to enable every built-in symbol preset

Symbol presets allow specific punctuation, operators, emoji, or math/science
symbols to pass the character gate without broadly accepting an entire language
script. For example, `--symbol-preset mathematics` can accept `π` while
`--keep-script Greek` would more broadly allow Greek-script text.

## Languages

The built-in dictionaries are loaded together by default:

- `english`: general English words
- `programming`: programming terms
- `linux`: Linux command names
- `api`: API and library names
- `icons`: icon-related terms

The bundled natural-language dictionary is currently English only. Additional
newline-separated word lists can be supplied by repeating `--dictionary`:

```bash
tokenpruner analyze \
  --model Qwen/Qwen3.5-Coder \
  --dictionary ./dictionaries/german.txt \
  --dictionary ./dictionaries/french.txt
```

`--wordfreq-lang` is separate from the bundled dictionaries and accepts the
following `wordfreq` language codes:

| Languages | Codes |
| --- | --- |
| Arabic, Bangla, Bosnian, Bulgarian, Catalan, Chinese | `ar`, `bn`, `bs`, `bg`, `ca`, `zh` |
| Croatian, Czech, Danish, Dutch, English, Finnish | `hr`, `cs`, `da`, `nl`, `en`, `fi` |
| French, German, Greek, Hebrew, Hindi, Hungarian | `fr`, `de`, `el`, `he`, `hi`, `hu` |
| Icelandic, Indonesian, Italian, Japanese, Korean, Latvian | `is`, `id`, `it`, `ja`, `ko`, `lv` |
| Lithuanian, Macedonian, Malay, Norwegian Bokmal, Persian, Polish | `lt`, `mk`, `ms`, `nb`, `fa`, `pl` |
| Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian | `pt`, `ro`, `ru`, `sr`, `sk`, `sl` |
| Spanish, Swedish, Tagalog, Tamil, Turkish, Ukrainian | `es`, `sv`, `fil`, `ta`, `tr`, `uk` |
| Urdu, Vietnamese | `ur`, `vi` |

Bosnian, Croatian, and Serbian share related `wordfreq` data; `sh` is also accepted
for the shared Serbo-Croatian list. Chinese, Japanese, and Korean may require the
optional `wordfreq[cjk]` dependencies. See the
[`wordfreq` supported-language documentation](https://github.com/rspeer/wordfreq#sources-and-supported-languages).

## Structural and Protected Tokens

- whitespace: spaces, tabs, newlines, and carriage returns
- control and format characters
- escape sequences such as `\n`, `\t`, `\xNN`, and `\uNNNN`
- tokenizer special tokens: BOS, EOS, padding, chat, and modality tokens
- byte-fallback tokens such as `<0x00>` through `<0xFF>`

## Accept/Prune Decision Tree

TokenPruner classifies each token with a deterministic decision tree. The final decision is binary:

- `accept`: keep this token as useful/protected for the target workload.
- `prune`: candidate for removal in a later pruning phase.

```mermaid
flowchart TD
    A[Tokenizer token] --> B[Decode token text]
    B --> C{Protected or structural?}
    C -->|yes| ACCEPT[accept]
    C -->|no| D{All characters allowed?}
    D -->|no| PRUNE[prune]
    D -->|yes| E[Split into word, symbol, and escape segments]

    E --> F{Has meaningful word segments?}
    F -->|no| G{All symbol/escape parts allowed?}
    G -->|yes| ACCEPT
    G -->|no| PRUNE

    F -->|yes| H["Segment and evaluate words:<br/>• lowercase and greedily split into longest dictionary words<br/>• also try capitalization boundaries, then greedy splitting<br/>• accept known words, complete greedy splits, and tokenizer fragments"]
    H --> J{Every word segment accepted?}
    J -->|yes| ACCEPT
    J -->|no| PRUNE
```

Examples:

- `中文,` -> `prune` if `Han` is not allowed; the comma does not rescue it.
- `classCallCheck` -> `accept` if `class`, `call`, and `check` are known.
- `typescript` -> `accept` if it fully splits into known words: `type` + `script`.
- `theros` -> `prune`; `the` is known, but `ros` is not, so the full token is not covered.
- `he` -> `accept` as a tokenizer fragment if it appears inside a known word such as `hello`.
- `{` -> `accept` with the `json` preset.
