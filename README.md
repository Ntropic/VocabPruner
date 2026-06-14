# TokenPruner

TokenPruner analyzes tokenizer vocabularies and decides which tokens to keep for a
target workload. It combines character scripts, symbol presets, dictionaries, and
`wordfreq` language evidence with a deterministic accept/prune decision tree.

## Install

```bash
conda activate tokenizer
pip install -e .
```

Shell completion:

```bash
eval "$(register-python-argcomplete tokenpruner)"
```

Hugging Face token:

```bash
echo 'export HF_TOKEN=<your_token>' >> ~/.bashrc
source ~/.bashrc
```

Create token: https://huggingface.co/settings/tokens/new

## Commands

- `inspect`: load a tokenizer/model config and report vocabulary metadata.
- `analyze`: classify vocabularies through the shared cache, print diagnostics, and create generated artifacts.
- `prune`: prune a supported model with compact text-token IDs; safetensors checkpoints are rewritten shard-by-shard without loading the full model.
- `benchmark`: compare a pruned model against its original model with token-count, optional quality, and optional speed checks.

## Parameters

- `--model`: Hugging Face model ID or local path.
- `--languages`: languages for which TokenPruner stores and checks frequency scores.
- `--wordfreq-min-zipf`: absolute word frequency cutoff; default/recommended threshold style.
- `--wordfreq-top-pct`: relative top-percent language vocabulary cutoff; alternative to Zipf.
- `--show`: display mode, e.g. `stats`, `all`, `examples`.
- `--create`: artifact creation mode, e.g. `list`, `decisions`, `html`.
- `--save [PATH]`: prune output path or model directory name; omit `PATH` to use a generated name under `pruned_models/`.
- `--workers`: process workers for token fact creation; cache reuse does not rebuild facts.

- `--keep-script`: allowed Unicode scripts for the character gate.
- `--symbol-preset`: allowed symbol families such as coding, JSON, mathematics, emoji.
- `--dictionary`: repeated custom term files for explicit domain/project vocabulary.
- `--seed`: deterministic `analyze` sampling/examples.
- `--trust-remote-code`: trust custom code from Hugging Face model repositories.
- `--yes`: skip prune confirmation prompts.


## Example Results

This run keeps Latin/Common/Inherited characters, all built-in symbol presets,
and English/German/Spanish word-frequency evidence at Zipf >= 3:

```bash
tokenpruner prune \
  --model inferRouter/Qwen3.6-27B-NVFP4 \
  --keep-script Latin Common Inherited \
  --symbol-preset all \
  --languages en de es \
  --wordfreq-min-zipf 3 \
  --show stats
```

For the 35B A3B run, use the same command and change only `--model` to
`nvidia/Qwen3.6-35B-A3B-NVFP4`.

Both tokenizers started with 248,077 tokens and retained 111,236 tokens, pruning
136,841 vocabulary entries. The checkpoint savings differ because each model
stores vocab-facing tensors differently.

| Model | Checkpoint Before | Checkpoint After | Saved | Checkpoint Reduction | Main-Path Bandwidth Saved | MTP-Path Bandwidth Saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inferRouter/Qwen3.6-27B-NVFP4` | 23.82 GB | 21.02 GB | 2.81 GB | 12% | 7% | 41% |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 23.42 GB | 22.70 GB | 0.72 GB | 3% | 5% | 38% |

The 35B model is A3B/MoE, so checkpoint size reduction and active weight
bandwidth reduction are different views of the same pruning result.

## Scripts

`--keep-script` accepts these script names:

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

`--languages` is separate from the bundled dictionaries and accepts the
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

TokenPruner first computes threshold-independent retain candidates, then applies
the selected wordfreq threshold. In cached facts, `retain_candidate` means the
token passed deterministic protection/script/symbol gates and is eligible for
language-frequency retention. It does not mean the token is definitely retained
for every Zipf or top-percent setting.

The final decision is binary:

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
