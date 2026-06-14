# `tokenpruner/hf_cache.py` ownership list

## Current functions/classes

- `CachedCheckpointStatus` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `repo_cache_dir` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `download_marker_path` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `read_download_marker` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `mark_analysis_download` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `mark_checkpoint_download` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `mark_download` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `cached_checkpoint_status` [keep] stay in `hf_cache.py`: HF cache marker/path logic belongs here
- `infer_cached_checkpoint_status` [delete] delete: disk inference fallback can accept partial downloads; completion marker is source of truth
- `snapshot_has_complete_safetensors` [delete] delete: disk inference fallback can accept partial downloads; completion marker is source of truth
- `snapshot_has_checkpoint_reference` [delete] delete: disk inference fallback can accept partial downloads; completion marker is source of truth
- `snapshot_has_analysis_files` [delete] delete: disk inference fallback can accept partial downloads; completion marker is source of truth

## Incoming functions/classes

- None
