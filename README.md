# VocabPruner

Shrink language models tokenizers to the vocabulary your workload actually needs.
VocabPruner helps you reduce the dictionary of your LLM's, to produce
smaller Hugging Face or GGUF checkpoints. Keep only the languages, writing systems,
symbols, and project-specific terms that matter to your workflows, while preserving
required encoding tokens. Smaller weights can leave more VRAM for KV cache;
quality and runtime effects should be measured on your workload.
VocabPruner supports standard language models, MTP models, and DFlash 1/2 and
DSpark 1/2 speculative decoding models.

## Install

```bash
pip install .
```

For development, use `pip install -e .` so edits under `src/vocabpruner/`
are available to the installed CLI. See the [Python API reference](docs/api.md)
for the package exports.

## Example
In the following example, `latin-all` keeps Latin text and a practical group
of useful symbols, including mathematical notation, emoji, ASCII punctuation, programming and data-format syntax. Tokens used only by unselected writing systems can be removed, except for protected encoding tokens and required merge dependencies.

```bash
vocabpruner prune \
  QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --selection-group latin-all \
  --save-suffix latin-all
```
The recorded measurements and output below predate the byte-token preservation
fix. Regenerate both target and draft checkpoints for current measurements.

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Vocabulary | 248,077 | 145,545 | −102,532 (−41.33%) |
| Checkpoint size | 18.36 GiB | 16.40 GiB | −1.96 GiB (−10.68%) |
| NVFP4 KV-cache capacity | — | — | ≈91,200 more tokens |

### Example Output

> **Pruned (`latin-all`)**
>
> **User:** Say "Hello World" in Japanese.
>
> **Assistant:** In Japanese, "Hello World" is "Hello, World" (Konnichiwa, Zaawoorudo).

> **Unpruned (original)**
>
> **User:** Say "Hello World" in Japanese.
>
> **Assistant:** In Japanese, "Hello World" is "こんにちは世界" (Konnichiwa, Zaawoorudo).

This historical output does not establish the behavior of newly pruned models.
Removing dedicated Japanese tokens does not remove the ability to encode Japanese
when byte encoding is preserved; it can require more tokens instead. Generated
answers and their quality can change even for retained scripts.

### Pruning Speculative Decoding Models
The pruned target directory contains the finalized old-to-new token-ID map.
Use `--related-model` to find that output by model name and apply its exact map
to a speculative decoding model that shares the target's original vocabulary:

```bash
vocabpruner prune \
  josch15366/Qwen3.8-27B-DFlash2-FP8 \
  --related-model Qwen3.8-27B-QUASAR-NVFP4-latin-all \
  --save-suffix latin-all
```

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Vocabulary | 248,320 | 145,545 | −102,775 (−41.39%) |
| Checkpoint size | 2.10 GiB | 2.00 GiB | −0.10 GiB (−4.68%) |

## CLI

```bash
vocabpruner COMMAND [MODEL] [PARAMETERS]
```

### Commands

| Command | Purpose |
| --- | --- |
| `init` | Create the optional user `.env` file. |
| `inspect [MODEL]` | Report vocabulary metadata. |
| `pull [MODEL]` | Download a model for local use. |
| `analyze [MODEL]` | Preview retained and removed tokens and create reports. |
| `prune [MODEL]` | Create a pruned Hugging Face or GGUF checkpoint. |

For commands that take `MODEL`, it is a Hugging Face model ID, local model
directory, or local `.gguf` file.

### Parameters

`analyze` and `prune` accept the parameters below.

#### Vocabulary Selection

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--keep-script` | See below | Preserve writing systems. |
| `--syntax-list` | See below | Preserve programming and data-format syntax. |
| `--symbol-list` | See below | Preserve symbol families. |
| `--languages` | See below | Preserve natural-language vocabulary. |
| `--selection-group` | See below | Apply grouped script, syntax, and symbol selections. |
| `--dictionary` | File path; repeatable | Preserve project or domain terms. |

#### Frequency Filtering

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--wordfreq-min-zipf` | Number | Keep words at or above an absolute frequency. |
| `--wordfreq-top-pct` | Percentage | Keep the most frequent percentage of words. |

If neither parameter is provided, word-frequency filtering is disabled.

#### Results

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--show` | See below | Print selected results. |
| `--create` | See below | Create artifacts without printing them. |
| `--save` | Optional path | Choose the pruned model path. |
| `--save-suffix` | Suffix | Replace the default `pruned` suffix. |
| `--related-path` | Pruned-model directory | Reuse its finalized token map. |
| `--related-model` | Model name or path | Reuse a named or explicit pruned-model directory. |

#### Run Control

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--workers` | Number | Set the number of analysis workers. |
| `--recreate-list` | Flag | Rebuild cached token data. |
| `--seed` | Number | Set deterministic sampling. |
| `--trust-remote-code` | Flag | Trust custom Hugging Face repository code. |
| `--yes` | Flag | Skip the confirmation prompt. |

### Options

The different parameters support a number of options listed below.

#### --keep-script

| Name | Examples | Size |
| --- | --- | ---: |
| `Latin` | A, é, ß, ... | 708 |
| `Greek` | α, β, Ω, ... | 400 |
| `Cyrillic` | Ж, Д, я, ... | 304 |
| `Arabic` | ا, ب, م, ... | 400 |
| `Hebrew` | א, ב, ש, ... | 112 |
| `Devanagari` | अ, क, म, ... | 128 |
| `Thai` | ก, ข, ม, ... | 128 |
| `Han` | 中, 文, 漢, ... | 28,096 |
| `Hiragana` | あ, い, ひ, ... | 96 |
| `Katakana` | ア, イ, カ, ... | 112 |
| `Hangul` | 한, 글, 가, ... | 11,536 |
| `Common` | 0, 1, separators¹, controls², ... | 5 |
| `Inherited` | combining acute ◌́, combining diaeresis ◌̈, ... | 1 |
| `all` | every script above |  |

¹ Separators are spaces and Unicode line or paragraph separators.
² Controls include tabs, newlines, control codes, formatting marks,
surrogates, private-use characters, and unassigned code points.

#### --syntax-list

| Name | Examples | Size |
| --- | --- | ---: |
| `coding` | {, }, =, ;, ... | 32 |
| `shell` | $, \|, >, \*, ... | 22 |
| `json` | {, }, :, true, false, null, ... | 23 |
| `yaml` | -, :, #, \|, ... | 20 |
| `xml` | &lt;, &gt;, /, =, ... | 15 |
| `markdown` | #, \*, \_, \`, ... | 16 |
| `all` | every syntax list above |  |

Counts include both characters and reserved words.

#### --symbol-list

| Name | Examples | Size |
| --- | --- | ---: |
| `mathematics` | +, =, ×, ∫, π, λ, Ω, ... | 74 |
| `ascii-punctuation` | !, ?, @, #, ... | 32 |
| `emoji` | 🙂, ✅, 🚀, ... | 3,523 |
| `all` | every symbol list above |  |

Symbol lists preserve specific punctuation, operators, emoji, or math/science
letters without keeping an entire script. For example, `--symbol-list
mathematics` can keep `π`, while `--keep-script Greek` keeps Greek text more
broadly.

Syntax and symbol entries participate in compositional matching. For example,
`hello+world` requires `+` from a selected list and both `hello` and `world`
to pass the selected language or dictionary rules.

#### --languages

Selects frequency-ranked natural-language vocabulary from `wordfreq`.

| Language | Code | Examples | Size |
| --- | --- | --- | ---: |
| English | `en` | the, to, and, ... | 28,917 |
| German | `de` | die, der, und, ... | 39,277 |
| Spanish | `es` | de, la, que, ... | 34,925 |
| French | `fr` | de, la, le, ... | 31,385 |
| Arabic | `ar` | في, من, على, ... | 56,642 |

These are the sizes of the `wordfreq` small lists used by VocabPruner. See
[`wordfreq`'s full list of supported languages and codes](https://github.com/rspeer/wordfreq#sources-and-supported-languages).

#### --selection-group

| Name | Scripts | Syntax Lists | Symbol Lists |
| --- | --- | --- | --- |
| `latin-all` | Latin, Common, Inherited | all | all |

Groups do not select languages or dictionaries. Add those separately when
needed.

#### --show

| Option | Purpose |
| --- | --- |
| `stats` | Summary counts. |
| `all` | Every token decision. |
| `examples` | Sample retained and pruned tokens. |
| `examples:N` | `N` retained and `N` pruned examples. |
| `examples:retained=N,pruned=N` | Separate example counts. |

#### --create

| Option | Purpose |
| --- | --- |
| `list` | Cached token facts. |
| `decisions` | Token decisions as JSONL. |
| `html` | Interactive word-frequency report. |

### --dictionary

Dictionary files are UTF-8 text with one term per line. Matching is
case-insensitive; blank lines and lines beginning with `#` are ignored.

```text
# Project terms
VocabPruner
tensor-parallel
NVFP4
```

Use relative or absolute paths and repeat `--dictionary` for multiple files:

```bash
--dictionary ./dictionaries/project.txt
--dictionary /absolute/path/to/domain.txt
```

### Editing Lists

List options come directly from JSON filenames in
`src/vocabpruner/resources/script_lists/`,
`src/vocabpruner/resources/syntax_lists/` and
`src/vocabpruner/resources/symbol_lists/`. Selection groups come from
`src/vocabpruner/resources/selection_groups/`. For example, adding `sql.json`
creates the `--syntax-list sql` option the next time VocabPruner starts. Each
syntax or symbol file uses this format:

```json
{
  "characters": "{}[]:,.\n",
  "words": ["true", "false", "null"],
  "ranges": [["U+1F000", "U+1FAFF"]]
}
```

`characters` is a string of individual characters, `words` contains complete
word matches, and `ranges` contains inclusive Unicode ranges. In JSON, `"\n"`
means a newline character while `"\\n"` means a backslash followed by `n`.
Changing a selected list automatically invalidates matching cached decisions.

Script lists use a similar JSON schema:

```json
{
  "characters": "",
  "categories": [],
  "ranges": [["U+0041", "U+005A"], ["U+0061", "U+007A"]]
}
```

`categories` contains Unicode general-category prefixes such as `M` for marks
or `P` for punctuation. Characters may match multiple script lists and pass
when any matching list is selected. Python's standard `unicodedata` module
provides the category assigned to each character.

Selection-group files combine existing lists without selecting languages or
dictionaries:

```json
{
  "keep_scripts": ["Latin", "Common", "Inherited"],
  "syntax_lists": ["all"],
  "symbol_lists": ["all"]
}
```

## Structural and Protected Tokens

- whitespace: spaces, tabs, newlines, and carriage returns
- control and format characters
- escape sequences such as `\n`, `\t`, `\xNN`, and `\uNNNN`
- tokenizer special tokens: BOS, EOS, padding, chat, and modality tokens
- byte-fallback tokens such as `<0x00>` through `<0xFF>`

Byte-level tokenizers also retain their complete 256-byte alphabet, including
printable byte symbols. BPE merge dependencies are retained recursively before
the token-ID map and model rows are rewritten. Old analysis caches are rebuilt
automatically when the cache schema changes.

### Tokenizer Support and Validation

Support is determined by the serialized tokenizer backend, **not the model family**.

| Tokenizer type / encoding | Support | Requirements and limits |
| --- | --- | --- |
| BPE | Supported | Hugging Face `tokenizer.json`; retained merge dependencies and merge order preserved. |
| Byte-level BPE | Supported | Complete 256-byte alphabet preserved when the backend declares ByteLevel preprocessing. |
| BPE / Unigram with byte fallback | Supported | All 256 `<0xNN>` fallback tokens required and preserved. |
| Unigram | Supported | JSON vocabulary, token scores, and unknown-token ID preserved/remapped. |
| WordPiece | Supported | JSON backend; `vocab.txt` also rewritten when present. |
| WordLevel / character vocabulary | Supported | Character vocabularies must use a supported backend, such as WordLevel. |
| SentencePiece-derived fast tokenizer | Conditional | Must expose supported BPE or Unigram JSON; native SentencePiece binary export is not supported. |
| GGUF tokenizer metadata | Limited validation | Stored vocabulary and metadata checked through our wrapper, not a native llama.cpp tokenizer. |
| Custom / unknown serialization | Unsupported unless recognized | Must expose one of the supported JSON backend formats. |

Format support does not guarantee that every model architecture or quantization
layout can be pruned. Tokenizers without byte encoding remain supported, but do
not gain universal byte coverage through pruning.

Hugging Face pruning supports JSON-backed BPE, Unigram, WordPiece, and WordLevel
tokenizers, including character vocabularies implemented with WordLevel.
Byte encoding is detected from tokenizer metadata, not the model-family name.
Tokenizers without byte fallback are allowed; pruning does not add that capability.
Custom implementations must expose one of these supported JSON backend formats.
Unknown serialization formats are refused rather than rewritten speculatively.

Every exported tokenizer is reloaded and checked against the source and the
old-to-new ID map. The check covers **every retained token**: its stored text,
individual decoding, special-token status, added-token attributes, and Unigram
score where applicable. BPE merge dependencies and merge order, unknown-token
IDs, normalization, preprocessing, decoding, and postprocessing are also checked.
Failure stops pruning; success writes `tokenizer_validation.json`.

These are tokenizer-file checks, not model inference or a proof that every input
will tokenize identically. Removed tokens can change segmentation. Individual
byte tokens may decode to a replacement character in both tokenizers; their
complete byte alphabet is verified separately. No API server or GPU is required
for the vocabulary check.

Use the exported Hugging Face tokenizer with `use_fast=True`. WordPiece
`vocab.txt` is remapped when present. Original SentencePiece binary sidecars are
removed from the export because they would retain the old vocabulary; the
validated `tokenizer.json` is the supported artifact.

GGUF exports verify retained token strings, IDs, scores, token types, byte-token
coverage, merges, and decoding through VocabPruner's GGUF wrapper, and write
`<checkpoint>.tokenizer_validation.json`. This is not a native llama.cpp
tokenization or inference test.

Previously pruned checkpoints are not repaired automatically. Regenerate them
from their original checkpoints, then regenerate related draft models using the
new target token map.

## Documentation

Documentation sources live in [`docs/`](docs/index.md). See the
[scoring reference](docs/scoring.md) for vocabulary selection rules.
The GitHub Actions workflow builds documentation on pull requests and
publishes from `main`; it does not release the package or upload models.

## Environment

VocabPruner loads the optional `~/.vocabpruner/.env` file. Create it with
commented examples when wanted:

```bash
vocabpruner init
```

Existing files are not overwritten. Values in this file override variables
inherited from the shell; explicit command-line options still take precedence.

| Variable | Purpose |
| --- | --- |
| `VOCABPRUNER_SAVE_DIR` | Default parent directory for pruned models. Defaults to the current directory. |
| `HF_HUB_CACHE` | Hugging Face model cache. |
| `HF_HOME` | Uses `<HF_HOME>/hub` when `HF_HUB_CACHE` is unset. |
| `XDG_CACHE_HOME` | Base cache directory when no Hugging Face cache variable is set; also controls the VocabPruner analysis cache on Linux. |


The analysis cache normally lives at
`${XDG_CACHE_HOME:-~/.cache}/vocabpruner` on Linux.
