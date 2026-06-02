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
