# VocabPruner

Shrink language models to the vocabulary your workload actually needs.
VocabPruner finds removable tokens, lets you review the result, and produces
smaller Hugging Face or GGUF checkpoints. Keep the languages, writing systems,
symbols, and project-specific terms that matter; leave the rest behind.

## Install

```bash
pip install .
```

## Example
As a useful example we keep only the latin alphabet and symbols for mathematical notation, emojis and coding.... 

### Unsloth NVFP4

```bash
vocabpruner prune \
  unsloth/Qwen3.8-27B-NVFP4 \
  --selection-group latin-all
```

Results for `Qwen3.8-27B-NVFP4` (LM head: FP8 with per column BF16 scaling):

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Vocabulary | 248,077 | 145,545 | −102,532 (−41.33%) |
| Checkpoint size | 21.81 GiB | 20.34 GiB | −1.47 GiB (−6.74%) |
| NVFP4 KV-cache capacity | — | — | ≈68,400 more tokens |

### QUASAR NVFP4

```bash
vocabpruner prune \
  QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --selection-group latin-all
```

Results for `Qwen3.8-27B-QUASAR-NVFP4` (LM head: BF16):

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Vocabulary | 248,077 | 145,545 | −102,532 (−41.33%) |
| Checkpoint size | 18.36 GiB | 16.40 GiB | −1.96 GiB (−10.68%) |
| NVFP4 KV-cache capacity | — | — | ≈91,200 more tokens |

## CLI

```bash
vocabpruner COMMAND MODEL [PARAMETERS]
```

### Commands

| Command | Purpose |
| --- | --- |
| `inspect` | Report vocabulary metadata. |
| `pull` | Download a model for local use. |
| `analyze` | Preview retained and removed tokens and create reports. |
| `prune` | Create a pruned Hugging Face or GGUF checkpoint. |

`MODEL` is a Hugging Face model ID, local model directory, or local `.gguf`
file.

### Model and Output Paths

Remote models use `--hf-cache PATH` when provided, followed by
`HF_HUB_CACHE`, `HF_HOME`, and Hugging Face's standard cache location. Local
models are used in place.

`--save PATH` sets the exact pruned-model directory. Without a path,
VocabPruner creates `<model-name>-pruned` under `VOCABPRUNER_SAVE_DIR`, or in
the current directory when that environment variable is unset. Generated
analysis data uses the operating system's user-cache directory.

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

#### Run Control

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--workers` | Number | Set the number of analysis workers. |
| `--hf-cache` | Path | Override the Hugging Face cache directory. |
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
`vocabpruner/resources/script_lists/`,
`vocabpruner/resources/syntax_lists/` and
`vocabpruner/resources/symbol_lists/`. Selection groups come from
`vocabpruner/resources/selection_groups/`. For example, adding `sql.json`
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
