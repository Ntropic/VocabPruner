# `tokenpruner/scoring.py` function merge matrix

| Function | Source | `ScoreConfig` | `score_token` | `decision_tree_status` | `_decision_explanation` | `CharacterGate` | `character_gate` | `is_structural_token` | `is_control_char` | `TokenSegment` | `token_segments` | `language_word_segments` | `language_evidence_segments` | `split_identifier_words` | `greedy_dictionary_split` | `_dictionary_split_terms` | `_greedy_split` | `is_word_char` | `has_language_evidence` | `Add function` | `Delete` | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ScoreConfig` | `tokenpruner/scoring.py` | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `score_token` | `tokenpruner/scoring.py` | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `decision_tree_status` | `tokenpruner/scoring.py` | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `_decision_explanation` | `tokenpruner/scoring.py` | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `CharacterGate` | `tokenpruner/scoring.py` | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `character_gate` | `tokenpruner/scoring.py` | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `is_structural_token` | `tokenpruner/scoring.py` | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `is_control_char` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `TokenSegment` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `token_segments` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `language_word_segments` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `language_evidence_segments` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `split_identifier_words` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `greedy_dictionary_split` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `_dictionary_split_terms` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `_greedy_split` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `is_word_char` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `has_language_evidence` | `tokenpruner/scoring.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | K | - | - | keep in `scoring.py`: scoring/config/token segmentation belongs here |
| `settings_config` | `tokenpruner/cli.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | M | - | move to `scoring.py`: scoring config policy belongs with ScoreConfig |
| `require_wordfreq_threshold` | `tokenpruner/cli.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | M | - | move to `scoring.py`: scoring config policy belongs with ScoreConfig |

Legend: `K` keep/stay, `M` move/add, `G` merge into existing function, `D` delete, `-` not applicable.
