# `tokenpruner/reports.py` ownership list

## Current functions/classes

- `ExampleCounts` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `render_jsonl` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `render_markdown` [delete] delete: unused markdown renderer
- `render_plain_text` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `filter_scores` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `should_show_rows` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `should_show_examples` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `normalize_show_options` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `score_stats` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `example_groups` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `_seeded_examples` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `_table` [delete] delete: helper only supports deleted markdown renderer
- `_plain_rows` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `_escape` [merge] merge into `reports.py::_rich_escape`: escaping helpers overlap; keep only renderer-needed escaping after markdown deletion
- `_rich_escape` [keep] stay in `reports.py`: report rendering/filtering helper belongs here
- `_show_label` [keep] stay in `reports.py`: report rendering/filtering helper belongs here

## Incoming functions/classes

- `format_gb` from `tokenpruner/cli.py` [move] move here: display formatting belongs with report/output helpers
