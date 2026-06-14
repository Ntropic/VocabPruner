# `tokenpruner/benchmark.py` function merge matrix

| Function | Source | `DatasetSpec` | `TokenDelta` | `SpeedResult` | `LmEvalRun` | `BenchmarkReport` | `load_prompt_file` | `load_dataset_prompts` | `dataset_spec` | `extract_text` | `append_text` | `measure_token_delta` | `token_count` | `run_lm_eval` | `measure_speed` | `write_report` | `report_to_dict` | `render_summary` | `model_slug` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DatasetSpec` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `TokenDelta` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `SpeedResult` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `LmEvalRun` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `BenchmarkReport` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `load_prompt_file` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `load_dataset_prompts` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `dataset_spec` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `extract_text` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `append_text` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `measure_token_delta` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `token_count` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `run_lm_eval` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `measure_speed` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `write_report` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `report_to_dict` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `render_summary` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: benchmark feature is out of cleanup scope unless explicitly accepted |
| `model_slug` | `tokenpruner/benchmark.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | G -> `pipeline.py::model_slug` | - | merge into `pipeline.py::model_slug`: duplicate slug helper; use pipeline slug policy, delete benchmark copy |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
