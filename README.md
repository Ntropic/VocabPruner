# TokenPruner

TokenPruner is a tokenizer vocabulary analysis toolkit. This first version focuses on
sample-first validation: it loads a tokenizer, samples vocabulary entries, scores only
those sampled tokens, and writes diagnostics that help evaluate whether the language,
script, symbol, icon, dictionary, and `wordfreq` signals are useful before full-vocab
pruning logic is implemented.

```bash
tokenpruner inspect --model Qwen/Qwen3.5-Coder

tokenpruner analyze \
  --model Qwen/Qwen3.5-Coder \
  --keep-script Latin Common Inherited \
  --drop-script Han Hiragana Katakana Hangul Cyrillic \
  --symbol-preset coding shell json markdown \
  --wordfreq-lang en \
  --sample-size 200 \
  --show stats ambiguous \
  --save ./reports/sampled_token_scores.jsonl
```

This project does not rewrite tokenizers, embeddings, model weights, or `lm_head`
matrices yet.

`inspect` also reads the model configuration and reports declared modality encoders,
vocabulary-bearing submodels, embedding/LM-head tying, and MTP settings. A separate
MTP head is only reported as confirmed when the configuration declares it; otherwise,
checkpoint-module inspection is still required before pruning.

## Supported Scripts and Languages

`--keep-script` and `--drop-script` accept these script names:

- `Latin`, `Greek`, `Cyrillic`
- `Arabic`, `Hebrew`, `Devanagari`, `Thai`
- `Han`, `Hiragana`, `Katakana`, `Hangul`
- `Common` for punctuation, symbols, numbers, separators, and controls
- `Inherited` for combining marks

Script selection controls which character families may pass the character gate. It
does not select a language dictionary. For example, `Latin` can contain English,
French, German, and many other languages.

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

Scores, if present in prototype output, are diagnostics only. They do not define the final decision.

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
