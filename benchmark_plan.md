# Benchmark plan

Status: planning-only. Benchmark implementation and tests were removed during
the dedupe cleanup; this document is a proposal for a future benchmark feature,
not a description of currently shipped code.

## Quality benchmarks
- `MMLU`: broad knowledge
- `HellaSwag`: commonsense completion
- `ARC-Easy`, `ARC-Challenge`: science QA
- `GSM8K`: math reasoning
- `TruthfulQA`: truthfulness
- `Winogrande`: commonsense pronoun/coreference
- `HumanEval`, `MBPP`: code generation

Runner: `lm-evaluation-harness` (`lm-eval`).

## Token-count datasets
Use small fixed samples. Measure `after_tokens / before_tokens`.

- `wikitext`: plain English
- `allenai/c4`: web text
- `EleutherAI/pile`: mixed pretraining text
- `cerebras/SlimPajama-627B`: cleaned pretraining mix
- `bigcode/the-stack` or `bigcode/the-stack-v2`: source code
- `irds/codesearchnet`: code + docstrings
- `tatsu-lab/alpaca`: instruction prompts
- `databricks/databricks-dolly-15k`: human instruction tuning
- `OpenAssistant/oasst1`: chat trees
- `HuggingFaceH4/ultrachat_200k`: chat/instruction
- `lmsys/lmsys-chat-1m`: real chat logs

## Fine-tune/test examples used by projects
- Stanford Alpaca: `tatsu-lab/alpaca`, 52K instruction examples
- Databricks Dolly: `databricks/databricks-dolly-15k`
- OpenAssistant: `OpenAssistant/oasst1`
- Meta Llama cookbook examples: Alpaca, grammar, SAMSum
- Code model work: `HumanEval`, `MBPP`, `CodeSearchNet`, `The Stack`

## Minimal module
- `tokenpruner/benchmark.py`
- functions:
  - `run_lm_eval(before, after, tasks)`
  - `measure_token_delta(before_tokenizer, after_tokenizer, datasets)`
  - `measure_speed(before, after, prompts)`
  - `write_report(path)`

## CLI
- Prune writes `tokenpruner_model.json` into pruned model dir.
- Benchmark can use only pruned model:
  - `tokenpruner benchmark --model pruned_models/MODEL --datasets wikitext alpaca`
- `--before-model/--after-model` still works for manual comparison.
- Pruned model args support `argcomplete` completion from `pruned_models/`.

## Metrics
- benchmark score before/after
- token count before/after
- token count ratio
- prompt throughput
- generation tokens/sec
- latency per prompt

## First default set
- Quality: `hellaswag,arc_easy,arc_challenge,mmlu,gsm8k,truthfulqa_mc2,winogrande`
- Token count: `wikitext,c4,alpaca,dolly,oasst1,codesearchnet,the-stack`
- Code quality: `humaneval,mbpp`

## Sources checked
- `lm-evaluation-harness`: task runner and task list
- Stanford Alpaca: 52K instruction-tuning examples
- Databricks Dolly 15K: human instruction data
- OpenAssistant OASST1: SFT/RM conversation trees
- BigCode The Stack: code pretraining corpus
- CodeSearchNet: code retrieval/docs corpus
