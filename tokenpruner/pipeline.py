from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tokenpruner.dictionaries import dictionary_matches, load_dictionaries
from tokenpruner.models import DictionaryMatch, ScriptProfile, TokenRecord, TokenScore, WordfreqMatch
from tokenpruner.scoring import (
    ScoreConfig,
    character_gate,
    has_language_evidence,
    is_structural_token,
    language_evidence_segments,
    language_word_segments,
    token_segments,
)
from tokenpruner.scripts import script_profile
from tokenpruner.symbols import matching_symbol_presets, symbol_presets
from tokenpruner.wordfreq_support import wordfreq_diagnostics

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PipelineResult:
    model: str
    records: list[TokenRecord]
    scores: list[TokenScore]
    retained_ids: list[int]
    pruned_ids: list[int]
    cache_dir: Path
    cache_reused: bool
    token_facts: dict[str, Any]
    wordfreq_facts: dict[str, Any]
    arrays: "PipelineArrays"


@dataclass(frozen=True)
class PipelineArrays:
    token_ids: np.ndarray
    retain_candidates: np.ndarray
    protected: np.ndarray
    wordfreq_token_ids: np.ndarray
    wordfreq_segment_indexes: np.ndarray
    language_ids: np.ndarray
    zipf: np.ndarray
    rank_percent: np.ndarray
    wordfreq_passes: np.ndarray
    best_zipf_by_segment: dict[tuple[int, int], float]
    best_rank_percent_by_segment: dict[tuple[int, int], float]
    segment_wordfreq_passes: dict[tuple[int, int], bool]


def should_show_html(show: list[str] | tuple[str, ...] | set[str]) -> bool:
    return "html" in set(show)


def write_wordfreq_html(cache_dir: Path, config: ScoreConfig) -> Path:
    wordfreq_path = cache_dir / "wordfreq_facts.json"
    policy_path = cache_dir / "policy.json"
    wordfreq_facts = json.loads(wordfreq_path.read_text(encoding="utf-8")) if wordfreq_path.exists() else {"columns": {}}
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    data = {
        "policy": policy,
        "threshold": {
            "wordfreq_min_zipf": config.wordfreq_min_zipf,
            "wordfreq_top_pct": config.wordfreq_top_pct,
        },
        "wordfreq": wordfreq_facts.get("columns", {}),
        "language_rows": sorted_language_wordfreq_rows(wordfreq_facts),
    }
    data_path = cache_dir / "wordfreq_data.js"
    html_path = cache_dir / "wordfreq_viewer.html"
    data_path.write_text(
        "window.TOKENPRUNER_WORDFREQ_DATA = "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    html_path.write_text(_wordfreq_html(), encoding="utf-8")
    return html_path


def _wordfreq_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TokenPruner Wordfreq Viewer</title>
  <script src="wordfreq_data.js"></script>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #101418; color: #f4f7fb; }
    header { display: flex; gap: 16px; align-items: center; padding: 14px 18px; border-bottom: 1px solid #2b333d; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    select { background: #19212a; color: #f4f7fb; border: 1px solid #536171; border-radius: 6px; padding: 6px 9px; }
    main { padding: 18px; }
    canvas { width: 100%; height: 560px; background: #151b22; border: 1px solid #2b333d; border-radius: 6px; display: block; }
    #tip { min-height: 28px; padding-top: 12px; color: #cbd5e1; font-size: 14px; }
  </style>
</head>
<body>
  <header>
    <h1>Wordfreq Distribution</h1>
    <select id="language"></select>
  </header>
  <main>
    <canvas id="chart"></canvas>
    <div id="tip"></div>
  </main>
  <script>
    const data = window.TOKENPRUNER_WORDFREQ_DATA || {wordfreq: {}, threshold: {}};
    const cols = data.wordfreq || {};
    const languages = [...new Set(cols.language || [])].sort();
    const select = document.getElementById('language');
    const canvas = document.getElementById('chart');
    const tip = document.getElementById('tip');
    const ctx = canvas.getContext('2d');

    for (const language of languages) {
      const option = document.createElement('option');
      option.value = language;
      option.textContent = language;
      select.append(option);
    }

    function rows(language) {
      return (data.language_rows && data.language_rows[language]) || [];
    }

    function pass(row) {
      if (data.threshold.wordfreq_top_pct != null) {
        return row.rankPercent != null && row.rankPercent <= data.threshold.wordfreq_top_pct;
      }
      return row.zipf >= data.threshold.wordfreq_min_zipf;
    }

    function draw() {
      const deviceRatio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * deviceRatio));
      canvas.height = Math.max(1, Math.floor(rect.height * deviceRatio));
      ctx.setTransform(deviceRatio, 0, 0, deviceRatio, 0, 0);
      const w = rect.width, h = rect.height, pad = 42;
      ctx.clearRect(0, 0, w, h);
      const list = rows(select.value);
      ctx.strokeStyle = '#334155';
      ctx.strokeRect(pad, pad, w - pad * 1.5, h - pad * 1.7);
      if (!list.length) {
        tip.textContent = 'No wordfreq rows for this language.';
        return;
      }
      const maxZipf = Math.max(...list.map(row => row.zipf), 1);
      const sampleCount = Math.min(600, list.length);
      ctx.beginPath();
      for (let i = 0; i < sampleCount; i++) {
        const source = Math.floor(i * (list.length - 1) / Math.max(1, sampleCount - 1));
        const x = pad + (i / Math.max(1, sampleCount - 1)) * (w - pad * 1.5);
        const y = h - pad * 0.7 - (list[source].zipf / maxZipf) * (h - pad * 1.7);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 2;
      ctx.stroke();
      tip.textContent = `${select.value}: ${list.length.toLocaleString()} rows`;
    }

    canvas.addEventListener('mousemove', event => {
      const list = rows(select.value);
      if (!list.length) return;
      const rect = canvas.getBoundingClientRect();
      const pad = 42;
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left - pad) / (rect.width - pad * 1.5)));
      const row = list[Math.min(list.length - 1, Math.max(0, Math.round(ratio * (list.length - 1))))];
      const rank = row.rank == null ? '-' : row.rank.toLocaleString();
      const pct = row.rankPercent == null ? '-' : `${row.rankPercent.toFixed(4)}%`;
      tip.textContent = `${row.term} | ${select.value} | zipf ${row.zipf.toFixed(3)} | rank ${rank} | top ${pct} | ${pass(row) ? 'passes' : 'fails'} | ${row.mode}`;
    });
    select.addEventListener('change', draw);
    window.addEventListener('resize', draw);
    draw();
  </script>
</body>
</html>
"""


def run_token_pipeline(
    *,
    model: str,
    config: ScoreConfig,
    cache_root: str | Path | None = None,
    records: list[TokenRecord],
    progress: bool = False,
) -> PipelineResult:
    cache_dir = token_cache_dir(model, config, cache_root=cache_root)
    policy = cache_policy(model, config)
    cached = load_cached_facts(cache_dir, policy)
    cache_reused = cached is not None
    if cached is None:
        token_facts, wordfreq_facts = build_token_facts(records, config, progress=progress)
        write_cached_facts(cache_dir, policy, token_facts, wordfreq_facts)
    else:
        token_facts, wordfreq_facts = cached
    arrays = arrays_from_facts(token_facts, wordfreq_facts, config)
    scores = derive_scores(records, token_facts, wordfreq_facts, config, arrays)
    retained_ids = [score.token.token_id for score in scores if score.status in {"protected", "retain_candidate"}]
    pruned_ids = [score.token.token_id for score in scores if score.status == "non_retain_candidate"]
    return PipelineResult(
        model=model,
        records=records,
        scores=scores,
        retained_ids=retained_ids,
        pruned_ids=pruned_ids,
        cache_dir=cache_dir,
        cache_reused=cache_reused,
        token_facts=token_facts,
        wordfreq_facts=wordfreq_facts,
        arrays=arrays,
    )


def token_cache_dir(model: str, config: ScoreConfig, cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else Path("tokenpruner_artifacts") / "cache"
    return root / compact_policy_slug(model, config, include_threshold=False)


def compact_policy_slug(model: str, config: ScoreConfig, include_threshold: bool = True) -> str:
    parts = [
        model_slug(model),
        f"scripts-{value_slug(sorted(config.keep_scripts))}",
        f"symbols-{symbol_slug(config.symbol_presets)}",
        f"langs-{value_slug(config.languages)}",
    ]
    if config.custom_dictionaries:
        parts.append(f"dicts-{value_slug(config.custom_dictionaries)}")
    if include_threshold:
        parts.append(threshold_slug(config))
    return "__".join(parts)


def cache_policy(model: str, config: ScoreConfig) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "policy": {
            "keep_scripts": sorted(config.keep_scripts),
            "symbol_presets": compact_symbol_values(config.symbol_presets),
            "languages": list(config.languages),
            "custom_dictionaries": list(config.custom_dictionaries),
            "builtin_dictionaries": (
                list(config.builtin_dictionaries)
                if config.builtin_dictionaries is not None
                else None
            ),
        },
    }


def load_cached_facts(cache_dir: Path, policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    policy_path = cache_dir / "policy.json"
    token_path = cache_dir / "token_facts.json"
    wordfreq_path = cache_dir / "wordfreq_facts.json"
    if not policy_path.exists() or not token_path.exists() or not wordfreq_path.exists():
        return None
    try:
        cached_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached_policy != policy:
        return None
    return (
        json.loads(token_path.read_text(encoding="utf-8")),
        json.loads(wordfreq_path.read_text(encoding="utf-8")),
    )


def write_cached_facts(
    cache_dir: Path,
    policy: dict[str, Any],
    token_facts: dict[str, Any],
    wordfreq_facts: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (cache_dir / "token_facts.json").write_text(json.dumps(token_facts, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (cache_dir / "wordfreq_facts.json").write_text(json.dumps(wordfreq_facts, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def build_token_facts(
    records: list[TokenRecord],
    config: ScoreConfig,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dictionaries = load_dictionaries(
        list(config.custom_dictionaries),
        list(config.builtin_dictionaries) if config.builtin_dictionaries is not None else None,
    )
    token_columns: dict[str, list[Any]] = {
        "token_id": [],
        "raw_token": [],
        "decoded_text": [],
        "is_special": [],
        "is_byte_fallback": [],
        "script_counts": [],
        "total_meaningful": [],
        "low_confidence": [],
        "protected": [],
        "gate_blocked": [],
        "retain_candidate": [],
        "has_word_segments": [],
        "evidence_segments": [],
        "dictionary_accepted": [],
        "symbol_presets": [],
        "dictionary_matches": [],
        "explanations": [],
    }
    wordfreq_columns: dict[str, list[Any]] = {
        "token_id": [],
        "segment_index": [],
        "language": [],
        "term": [],
        "mode": [],
        "zipf": [],
        "rank": [],
        "rank_percent": [],
    }
    iterable = _progress(records, progress=progress, total=len(records))
    for record in iterable:
        fact = deterministic_token_fact(record, config, dictionaries)
        for key, value in fact.items():
            token_columns[key].append(value)
        if fact["retain_candidate"] and not fact["protected"]:
            for segment_index, segment in enumerate(fact["evidence_segments"]):
                for diagnostic in wordfreq_diagnostics(segment, list(config.languages)):
                    wordfreq_columns["token_id"].append(record.token_id)
                    wordfreq_columns["segment_index"].append(segment_index)
                    wordfreq_columns["language"].append(diagnostic.language)
                    wordfreq_columns["term"].append(diagnostic.term)
                    wordfreq_columns["mode"].append(diagnostic.mode)
                    wordfreq_columns["zipf"].append(diagnostic.frequency)
                    wordfreq_columns["rank"].append(diagnostic.rank)
                    wordfreq_columns["rank_percent"].append(diagnostic.rank_percent)
    return (
        {"schema_version": CACHE_SCHEMA_VERSION, "columns": token_columns},
        {"schema_version": CACHE_SCHEMA_VERSION, "columns": wordfreq_columns},
    )


def deterministic_token_fact(record: TokenRecord, config: ScoreConfig, dictionaries: dict[str, set[str]]) -> dict[str, Any]:
    text = record.decoded_text or record.raw_token
    profile = script_profile(text)
    structural = is_structural_token(text)
    protected = record.is_special or record.is_byte_fallback or structural
    segments = token_segments(text)
    gate = character_gate(text, config, segments)
    presets = matching_symbol_presets(text, config.symbol_presets)
    word_segments = language_word_segments(segments)
    evidence_segments = language_evidence_segments(word_segments, dictionaries)
    dict_matches_by_segment = [dictionary_matches(segment.text, dictionaries) for segment in evidence_segments]
    dictionary_accepted = [
        has_language_evidence(segment.text, segment_matches, [])
        for segment, segment_matches in zip(evidence_segments, dict_matches_by_segment)
    ]
    explanations: list[str] = []
    if record.is_special:
        explanations.append("special token is always preserved")
    if record.is_byte_fallback:
        explanations.append("byte fallback token is always preserved")
    if structural:
        explanations.append("decoded whitespace/control token is always preserved")
    if not protected and gate.blocked:
        explanations.append(f"character gate blocked disallowed character families: {', '.join(gate.blocked)}")
    elif not protected:
        explanations.append("character gate passed")
    if presets and (protected or not gate.blocked):
        explanations.append(f"symbol/icon presets matched: {', '.join(sorted(set(presets)))}")
    flat_dict_matches = [match.to_dict() for matches in dict_matches_by_segment for match in matches[:10]]
    if flat_dict_matches:
        top = ", ".join(f"{match['dictionary']}:{match['term']}/{match['mode']}" for match in flat_dict_matches[:3])
        explanations.append(f"dictionary evidence matched: {top}")
    if not protected and not gate.blocked and not word_segments and presets:
        explanations.append("symbol-only token retained by selected symbol presets")
    retain_candidate = protected or not gate.blocked
    return {
        "token_id": record.token_id,
        "raw_token": record.raw_token,
        "decoded_text": record.decoded_text,
        "is_special": record.is_special,
        "is_byte_fallback": record.is_byte_fallback,
        "script_counts": dict(profile.counts),
        "total_meaningful": profile.total_meaningful,
        "low_confidence": profile.low_confidence,
        "protected": protected,
        "gate_blocked": list(gate.blocked),
        "retain_candidate": retain_candidate,
        "has_word_segments": bool(word_segments),
        "evidence_segments": [segment.text for segment in evidence_segments],
        "dictionary_accepted": dictionary_accepted,
        "symbol_presets": sorted(set(presets)),
        "dictionary_matches": flat_dict_matches[:10],
        "explanations": explanations,
    }


def arrays_from_facts(token_facts: dict[str, Any], wordfreq_facts: dict[str, Any], config: ScoreConfig) -> PipelineArrays:
    token_columns = token_facts["columns"]
    wordfreq_columns = wordfreq_facts["columns"]
    language_to_id = {language: index for index, language in enumerate(config.languages)}
    rank_percent = [value if value is not None else np.inf for value in wordfreq_columns["rank_percent"]]
    zipf = np.asarray(wordfreq_columns["zipf"], dtype=np.float64)
    rank_percent_array = np.asarray(rank_percent, dtype=np.float64)
    if config.wordfreq_top_pct is not None:
        wordfreq_passes = rank_percent_array <= config.wordfreq_top_pct
    else:
        wordfreq_passes = zipf >= config.wordfreq_min_zipf
    token_ids = np.asarray(wordfreq_columns["token_id"], dtype=np.int64)
    segment_indexes = np.asarray(wordfreq_columns["segment_index"], dtype=np.int64)
    best_zipf = _best_by_segment(token_ids, segment_indexes, zipf, reducer=max)
    best_rank_percent = _best_by_segment(token_ids, segment_indexes, rank_percent_array, reducer=min)
    segment_passes = _segment_passes(token_ids, segment_indexes, wordfreq_passes)
    return PipelineArrays(
        token_ids=np.asarray(token_columns["token_id"], dtype=np.int64),
        retain_candidates=np.asarray(token_columns["retain_candidate"], dtype=bool),
        protected=np.asarray(token_columns["protected"], dtype=bool),
        wordfreq_token_ids=token_ids,
        wordfreq_segment_indexes=segment_indexes,
        language_ids=np.asarray([language_to_id.get(language, -1) for language in wordfreq_columns["language"]], dtype=np.int64),
        zipf=zipf,
        rank_percent=rank_percent_array,
        wordfreq_passes=wordfreq_passes,
        best_zipf_by_segment=best_zipf,
        best_rank_percent_by_segment=best_rank_percent,
        segment_wordfreq_passes=segment_passes,
    )


def sorted_language_wordfreq_rows(wordfreq_facts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    columns = wordfreq_facts.get("columns", {})
    languages = np.asarray(columns.get("language", []), dtype=object)
    terms = np.asarray(columns.get("term", []), dtype=object)
    modes = np.asarray(columns.get("mode", []), dtype=object)
    zipf = np.asarray(columns.get("zipf", []), dtype=np.float64)
    ranks = np.asarray(columns.get("rank", []), dtype=object)
    rank_percent = np.asarray(columns.get("rank_percent", []), dtype=object)
    rows: dict[str, list[dict[str, Any]]] = {}
    for language in sorted(set(str(value) for value in languages)):
        indexes = np.flatnonzero(languages == language)
        sorted_indexes = indexes[np.argsort(zipf[indexes])[::-1]]
        rows[language] = [
            {
                "term": str(terms[index]),
                "mode": str(modes[index]),
                "zipf": float(zipf[index]),
                "rank": None if ranks[index] is None else int(ranks[index]),
                "rankPercent": None if rank_percent[index] is None else float(rank_percent[index]),
            }
            for index in sorted_indexes
        ]
    return rows


def _best_by_segment(token_ids: np.ndarray, segment_indexes: np.ndarray, values: np.ndarray, reducer) -> dict[tuple[int, int], float]:
    best: dict[tuple[int, int], float] = {}
    for token_id, segment_index, value in zip(token_ids, segment_indexes, values):
        if np.isinf(value):
            continue
        key = (int(token_id), int(segment_index))
        best[key] = float(value) if key not in best else float(reducer(best[key], float(value)))
    return best


def _segment_passes(token_ids: np.ndarray, segment_indexes: np.ndarray, passes: np.ndarray) -> dict[tuple[int, int], bool]:
    passed: dict[tuple[int, int], bool] = {}
    for token_id, segment_index, does_pass in zip(token_ids, segment_indexes, passes):
        key = (int(token_id), int(segment_index))
        passed[key] = passed.get(key, False) or bool(does_pass)
    return passed


def derive_scores(
    records: list[TokenRecord],
    token_facts: dict[str, Any],
    wordfreq_facts: dict[str, Any],
    config: ScoreConfig,
    arrays: PipelineArrays,
) -> list[TokenScore]:
    token_columns = token_facts["columns"]
    wordfreq_by_token_segment: dict[tuple[int, int], list[dict[str, Any]]] = {}
    wordfreq_columns = wordfreq_facts["columns"]
    for index, token_id in enumerate(wordfreq_columns["token_id"]):
        key = (token_id, wordfreq_columns["segment_index"][index])
        wordfreq_by_token_segment.setdefault(key, []).append({column: values[index] for column, values in wordfreq_columns.items()})
    records_by_id = {record.token_id: record for record in records}
    scores: list[TokenScore] = []
    for index, token_id in enumerate(token_columns["token_id"]):
        record = records_by_id.get(token_id) or TokenRecord(
            token_id=token_id,
            raw_token=token_columns["raw_token"][index],
            decoded_text=token_columns["decoded_text"][index],
            is_special=token_columns["is_special"][index],
            is_byte_fallback=token_columns["is_byte_fallback"][index],
        )
        protected = bool(token_columns["protected"][index])
        retain_candidate = bool(token_columns["retain_candidate"][index])
        evidence_segments = list(token_columns["evidence_segments"][index])
        dictionary_accepted = list(token_columns["dictionary_accepted"][index])
        explanations = list(token_columns["explanations"][index])
        passing_wordfreq: list[WordfreqMatch] = []
        segment_acceptance: list[bool] = []
        for segment_index, segment in enumerate(evidence_segments):
            accepted = bool(dictionary_accepted[segment_index])
            diagnostics = wordfreq_by_token_segment.get((token_id, segment_index), [])
            segment_wordfreq_passes = arrays.segment_wordfreq_passes.get((int(token_id), segment_index), False)
            if segment_wordfreq_passes:
                segment_matches = [_diagnostic_to_match(item) for item in diagnostics if _diagnostic_passes_threshold(item, config)]
                accepted = True
                passing_wordfreq.extend(segment_matches)
            segment_acceptance.append(accepted)
        if passing_wordfreq:
            passing_wordfreq.sort(key=lambda match: match.frequency, reverse=True)
            top = ", ".join(f"{match.language}:{match.term}/{match.mode}" for match in passing_wordfreq[:3])
            explanations.append(f"wordfreq evidence matched: {top}")

        if protected:
            status = "protected"
            explanations.append("decision tree: accept protected token")
        elif not retain_candidate:
            status = "non_retain_candidate"
            explanations.append("decision tree: prune because the character gate failed")
        elif not evidence_segments:
            status = "retain_candidate"
            explanations.append("decision tree: accept allowed symbol, escape, or short-fragment token")
        elif all(segment_acceptance):
            status = "retain_candidate"
            explanations.append("decision tree: accept because every word segment was accepted")
        else:
            status = "non_retain_candidate"
            missing = [segment for segment, accepted in zip(evidence_segments, segment_acceptance) if not accepted]
            if missing:
                explanations.append(f"word segment(s) lack language evidence: {', '.join(missing[:5])}")
            explanations.append("decision tree: prune because at least one word segment was not accepted")

        scores.append(
            TokenScore(
                token=record,
                script_profile=ScriptProfile(
                    counts=dict(token_columns["script_counts"][index]),
                    total_meaningful=token_columns["total_meaningful"][index],
                    low_confidence=token_columns["low_confidence"][index],
                ),
                protected=protected,
                status=status,
                symbol_presets=list(token_columns["symbol_presets"][index]),
                dictionary_matches=[DictionaryMatch(**match) for match in token_columns["dictionary_matches"][index]],
                wordfreq_matches=passing_wordfreq[:10],
                explanations=explanations,
            )
        )
    return sorted(scores, key=lambda score: score.token.token_id)


def _diagnostic_to_match(item: dict[str, Any]) -> WordfreqMatch:
    return WordfreqMatch(
        language=item["language"],
        term=item["term"],
        mode=item["mode"],
        frequency=item["zipf"],
        rank=item["rank"],
        rank_percent=item["rank_percent"],
    )


def _diagnostic_passes_threshold(item: dict[str, Any], config: ScoreConfig) -> bool:
    if config.wordfreq_top_pct is not None:
        rank_percent = item["rank_percent"]
        return rank_percent is not None and rank_percent <= config.wordfreq_top_pct
    return item["zipf"] >= config.wordfreq_min_zipf


def compact_symbol_values(values: set[str]) -> list[str]:
    return ["all"] if set(values) == set(symbol_presets()) else sorted(values)


def symbol_slug(values: set[str]) -> str:
    return value_slug(compact_symbol_values(values))


def threshold_slug(config: ScoreConfig) -> str:
    if config.wordfreq_top_pct is not None:
        return f"top-pct-{number_slug(config.wordfreq_top_pct)}"
    return f"zipf-{number_slug(config.wordfreq_min_zipf)}"


def value_slug(values) -> str:
    value_list = list(values)
    if not value_list:
        return "none"
    return "-".join(model_slug(str(value)) for value in sorted(value_list))


def number_slug(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def model_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    slug = slug.strip("-").lower()
    return slug or "unnamed"


def _progress(iterable, progress: bool, total: int):
    if not progress:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="Classifying token facts", unit="tok")
