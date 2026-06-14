# `tokenpruner/cli.py` ownership list

## Current functions/classes

- `default_workers` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `build_parser` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `add_settings_arguments` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `add_worker_argument` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `add_save_argument` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `add_show_argument` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `add_create_argument` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `inspect_command` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `analyze_command` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `prune_command` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `main` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `app` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `settings_config` [move] move to `scoring.py`: scoring config policy belongs with ScoreConfig
- `require_wordfreq_threshold` [move] move to `scoring.py`: scoring config policy belongs with ScoreConfig
- `cached_create_only_pipeline` [move] move to `pipeline.py`: cache/pipeline construction belongs with pipeline
- `write_token_decisions` [move] move to `pipeline.py`: list/artifact writing belongs with pipeline outputs until new module approved
- `write_create_artifacts` [move] move to `pipeline.py`: list/artifact writing belongs with pipeline outputs until new module approved
- `start_html_server` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `free_local_port` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `score_config_to_dict` [delete] delete: unused helper
- `resolve_save_dir` [merge] merge into `pipeline.py::write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `decision_artifact_dir` [merge] merge into `pipeline.py::write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `format_gb` [move] move to `reports.py`: display formatting belongs with report/output helpers
- `print_pruning_preview` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `print_cache_status` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `confirm_pruning` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `ensure_pruning_runtime` [move] move to `pruning.py`: runtime dependency policy belongs with pruning
- `print_prune_decisions` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `normalize_show_args` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `normalize_create_args` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here
- `should_create_list` [merge] merge into `pipeline.py::write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `should_create_decisions` [merge] merge into `pipeline.py::write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `should_create_html` [merge] merge into `pipeline.py::write_create_artifacts`: artifact path/selection helpers overlap output-writing workflow; unify there
- `parse_examples_spec` [keep] stay in `cli.py`: CLI parser, dispatch, prompt, local server, or terminal printing belongs here

## Incoming functions/classes

- None
