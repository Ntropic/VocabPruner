# VocabPruner
I don't speak chinese, so why waste precious memory on it in my LLM's. 
Prune LLM vocabularies and model weights by script, language, and symbols. 
 - Supports Hugging Face and GGUF checkpoints
 - MTP, DFlash, and DSpark models
 - Preserves byte encoding
 - Supports BPE, Unigram, WordPiece, and WordLevel tokenizers
 - Supports wordfrequency based pruning

## Install

```bash
git clone https://github.com/Ntropic/VocabPruner.git
pip install ./VocabPruner
```

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
While all of the remaining tokens retain their relative probability scores, removed tokens can lead to unusual results.  
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

### Tokenizer Validation

Hugging Face exports require a supported JSON-backed fast tokenizer.
SentencePiece-derived BPE and Unigram are supported through `tokenizer.json`;
native SentencePiece binary export is not supported.

Every retained token is checked against the source for representation, decoding,
and ID mapping, alongside tokenizer settings and BPE dependencies. Failures stop
pruning; results are saved in `tokenizer_validation.json`. Use `use_fast=True`.

GGUF checks use our tokenizer wrapper. These checks verify tokenizer integrity;
they do not guarantee identical segmentation or model output.

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
