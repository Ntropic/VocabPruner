# `tokenpruner/hf_cache.py` function merge matrix

| Function | Source | `CachedCheckpointStatus` | `repo_cache_dir` | `download_marker_path` | `read_download_marker` | `mark_analysis_download` | `mark_checkpoint_download` | `mark_download` | `cached_checkpoint_status` | `infer_cached_checkpoint_status` | `snapshot_has_complete_safetensors` | `snapshot_has_checkpoint_reference` | `snapshot_has_analysis_files` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CachedCheckpointStatus` | `tokenpruner/hf_cache.py` | K | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `repo_cache_dir` | `tokenpruner/hf_cache.py` | - | K | - | - | - | - | - | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `download_marker_path` | `tokenpruner/hf_cache.py` | - | - | K | - | - | - | - | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `read_download_marker` | `tokenpruner/hf_cache.py` | - | - | - | K | - | - | - | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `mark_analysis_download` | `tokenpruner/hf_cache.py` | - | - | - | - | K | - | - | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `mark_checkpoint_download` | `tokenpruner/hf_cache.py` | - | - | - | - | - | K | - | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `mark_download` | `tokenpruner/hf_cache.py` | - | - | - | - | - | - | K | - | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `cached_checkpoint_status` | `tokenpruner/hf_cache.py` | - | - | - | - | - | - | - | K | - | - | - | - | - | - | keep in `hf_cache.py`: HF cache marker/path logic belongs here |
| `infer_cached_checkpoint_status` | `tokenpruner/hf_cache.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: disk inference fallback can accept partial downloads; completion marker is source of truth |
| `snapshot_has_complete_safetensors` | `tokenpruner/hf_cache.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: disk inference fallback can accept partial downloads; completion marker is source of truth |
| `snapshot_has_checkpoint_reference` | `tokenpruner/hf_cache.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: disk inference fallback can accept partial downloads; completion marker is source of truth |
| `snapshot_has_analysis_files` | `tokenpruner/hf_cache.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | D | delete: disk inference fallback can accept partial downloads; completion marker is source of truth |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
