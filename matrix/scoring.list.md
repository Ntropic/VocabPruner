# `tokenpruner/scoring.py` ownership list

## Current functions/classes

- `ScoreConfig` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `score_token` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `decision_tree_status` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `_decision_explanation` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `CharacterGate` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `character_gate` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `is_structural_token` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `is_control_char` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `TokenSegment` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `token_segments` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `language_word_segments` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `language_evidence_segments` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `split_identifier_words` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `greedy_dictionary_split` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `_dictionary_split_terms` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `_greedy_split` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `is_word_char` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here
- `has_language_evidence` [keep] stay in `scoring.py`: scoring/config/token segmentation belongs here

## Incoming functions/classes

- `settings_config` from `tokenpruner/cli.py` [move] move here: scoring config policy belongs with ScoreConfig
- `require_wordfreq_threshold` from `tokenpruner/cli.py` [move] move here: scoring config policy belongs with ScoreConfig
