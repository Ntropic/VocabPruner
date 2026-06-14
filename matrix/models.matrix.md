# `tokenpruner/models.py` function merge matrix

| Function | Source | `TokenRecord` | `ScriptProfile` | `DictionaryMatch` | `WordfreqMatch` | `TokenScore` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `TokenRecord` | `tokenpruner/models.py` | K | - | - | - | - | - | - | keep in `models.py`: shared data model belongs here |
| `ScriptProfile` | `tokenpruner/models.py` | - | K | - | - | - | - | - | keep in `models.py`: shared data model belongs here |
| `DictionaryMatch` | `tokenpruner/models.py` | - | - | K | - | - | - | - | keep in `models.py`: shared data model belongs here |
| `WordfreqMatch` | `tokenpruner/models.py` | - | - | - | K | - | - | - | keep in `models.py`: shared data model belongs here |
| `TokenScore` | `tokenpruner/models.py` | - | - | - | - | K | - | - | keep in `models.py`: shared data model belongs here |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
