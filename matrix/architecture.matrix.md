# `tokenpruner/architecture.py` function merge matrix

| Function | Source | `ModelArchitectureReport` | `load_model_config` | `inspect_model_architecture` | `render_model_architecture` | `_flatten` | `_encoder_paths` | `_is_vocabulary_size` | `_is_mtp_setting` | `_is_mtp_path` | `_enabled` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ModelArchitectureReport` | `tokenpruner/architecture.py` | K | - | - | - | - | - | - | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `load_model_config` | `tokenpruner/architecture.py` | - | K | - | - | - | - | - | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `inspect_model_architecture` | `tokenpruner/architecture.py` | - | - | K | - | - | - | - | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `render_model_architecture` | `tokenpruner/architecture.py` | - | - | - | K | - | - | - | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `_flatten` | `tokenpruner/architecture.py` | - | - | - | - | K | - | - | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `_encoder_paths` | `tokenpruner/architecture.py` | - | - | - | - | - | K | - | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `_is_vocabulary_size` | `tokenpruner/architecture.py` | - | - | - | - | - | - | K | - | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `_is_mtp_setting` | `tokenpruner/architecture.py` | - | - | - | - | - | - | - | K | - | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `_is_mtp_path` | `tokenpruner/architecture.py` | - | - | - | - | - | - | - | - | K | - | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |
| `_enabled` | `tokenpruner/architecture.py` | - | - | - | - | - | - | - | - | - | K | - | - | keep in `architecture.py`: architecture inspection/config helper belongs here |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
