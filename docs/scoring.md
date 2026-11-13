# Scoring system

Selection uses a decision tree, not an aggregate numeric quality score.
It evaluates decoded token text, falling back to the stored token when empty.

## Decision order

1. Preserve special tokens, recognized byte tokens, and structural whitespace
   or control tokens as `protected`.
2. Reject disallowed character families as `non_retain_candidate`.
3. With frequency filtering disabled, accept tokens passing the character gate
   as `retain_candidate`.
4. Otherwise accept tokens without language-evidence segments. For tokens with
   evidence segments, every segment needs accepted dictionary, syntax-word, or
   frequency evidence.

## Character selection

Script membership comes from packaged lists. Unmatched characters are `Unknown`.
A character passes when a selected script or syntax/symbol list allows it.
Common punctuation and symbols require list membership. Whitespace, controls,
and recognized escape sequences bypass the gate.

`latin-all` selects Latin, Common, and Inherited plus all syntax/symbol lists;
it does not select all writing systems.

## Language evidence

Word segments are expanded using dictionary splitting and identifier
capitalization boundaries. Selected syntax words and dictionary matches can
provide evidence. Frequency evidence uses configured `wordfreq` languages with
either a minimum Zipf frequency or a top-ranked percentage. Thresholds are
mutually exclusive. Any selected language can accept a segment; every evidence
segment must pass.

The CLI disables frequency filtering unless a threshold is supplied.
`--languages en` alone therefore does not restrict vocabulary to English.

## Preservation and validation

Pruning supplements retained candidates with required byte tokens, recursive BPE
merge dependencies, and required model configuration IDs. These may include
entries outside selected scripts.

Export validation checks retained vocabulary entries and tokenizer processing
settings against the source. It does not measure model quality or guarantee
identical segmentation or generated responses.
