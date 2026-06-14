# `tokenpruner/reports.py` function merge matrix

| Function | Source | `ExampleCounts` | `render_jsonl` | `render_markdown` | `render_plain_text` | `filter_scores` | `should_show_rows` | `should_show_examples` | `normalize_show_options` | `score_stats` | `example_groups` | `_seeded_examples` | `_table` | `_plain_rows` | `_escape` | `_rich_escape` | `_show_label` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ExampleCounts` | `tokenpruner/reports.py` | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `render_jsonl` | `tokenpruner/reports.py` | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `render_markdown` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: unused markdown renderer |
| `render_plain_text` | `tokenpruner/reports.py` | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `filter_scores` | `tokenpruner/reports.py` | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `should_show_rows` | `tokenpruner/reports.py` | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `should_show_examples` | `tokenpruner/reports.py` | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `normalize_show_options` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `score_stats` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `example_groups` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `_seeded_examples` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `_table` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: helper only supports deleted markdown renderer |
| `_plain_rows` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `_escape` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | G | - | - | - | merge into `reports.py::_rich_escape`: escaping helpers overlap; keep only renderer-needed escaping after markdown deletion |
| `_rich_escape` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `_show_label` | `tokenpruner/reports.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | keep in `reports.py`: report rendering/filtering helper belongs here |
| `format_gb` | `tokenpruner/cli.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | M | - | move to `reports.py`: display formatting belongs with report/output helpers |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
