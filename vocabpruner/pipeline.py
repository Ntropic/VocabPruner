from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vocabpruner.dictionaries import dictionary_matches, load_dictionaries
from vocabpruner.models import DictionaryMatch, ScriptProfile, TokenRecord, TokenScore, WordfreqMatch
from vocabpruner.paths import DECISIONS_DIR, LISTS_DIR, pruned_models_dir
from vocabpruner.reports import render_jsonl
from vocabpruner.scoring import (
    ScoreConfig,
    character_gate,
    has_language_evidence,
    is_structural_token,
    language_evidence_segments,
    language_word_segments,
    token_segments,
)
from vocabpruner.scripts import compact_script_values, script_list_fingerprint, script_profile
from vocabpruner.symbols import (
    compact_list_values,
    matching_symbol_lists,
    matching_syntax_lists,
    matching_syntax_word_lists,
    symbol_list_fingerprint,
    symbol_lists,
    syntax_list_fingerprint,
    syntax_lists,
)
from vocabpruner.wordfreq_support import wordfreq_diagnostics

CACHE_SCHEMA_VERSION = 4


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
    arrays: "PipelineArrays | None"


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


def write_wordfreq_html(
    cache_dir: Path,
    config: ScoreConfig,
    output_dir: str | Path | None = None,
    *,
    wordfreq_min_zipf: float | None = None,
    wordfreq_top_pct: float | None = None,
) -> Path:
    destination = Path(output_dir) if output_dir is not None else cache_dir
    destination.mkdir(parents=True, exist_ok=True)
    html_path = destination / "wordfreq_viewer.html"
    html_path.write_text(
        _wordfreq_html(
            settings_path=_viewer_relative_path(destination, cache_dir / "settings.json"),
            token_path=_viewer_relative_path(destination, cache_dir / "token_facts.json"),
            wordfreq_path=_viewer_relative_path(destination, cache_dir / "wordfreq_facts.json"),
        ),
        encoding="utf-8",
    )
    return html_path


def write_token_decisions(save: str | Path, scores) -> Path:
    output_dir = Path(save)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "token_decisions.jsonl"
    path.write_text(render_jsonl(scores) + ("\n" if scores else ""), encoding="utf-8")
    return path


def write_create_artifacts(
    create: list[str],
    output_dir: str | Path,
    *,
    config: ScoreConfig,
    scores=None,
    cache_dir: Path | None = None,
    html_dir: str | Path | None = None,
    wordfreq_min_zipf: float | None = None,
    wordfreq_top_pct: float | None = None,
    html_launcher: Callable[[Path], tuple[str, int]] | None = None,
) -> Path:
    resolved_output_dir = Path(output_dir)
    if not create:
        return resolved_output_dir
    if cache_dir is None:
        raise ValueError("cache_dir is required when creating artifacts")
    html_output_dir = Path(html_dir) if html_dir is not None else resolved_output_dir
    if "list" in create:
        print(f"List: {cache_dir}", file=sys.stderr)
    if "decisions" in create:
        decisions_path = write_token_decisions(resolved_output_dir, scores or [])
        print(f"Decisions: {decisions_path}", file=sys.stderr)
    if "html" in create:
        html_path = write_wordfreq_html(
            cache_dir,
            config,
            output_dir=html_output_dir,
            wordfreq_min_zipf=wordfreq_min_zipf,
            wordfreq_top_pct=wordfreq_top_pct,
        )
        print(f"HTML: {html_path}")
        if html_launcher is not None:
            url, pid = html_launcher(html_output_dir)
            print(f"Open: {url}")
            print(f"Server PID: {pid}", file=sys.stderr)
    return resolved_output_dir


def resolve_artifact_dir(
    output_dir: str | Path | None,
    *,
    command: str,
    model: str,
    config: ScoreConfig,
    seed: int | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    if command == "prune":
        model_name = Path(model).name
        if model_name.lower().endswith(".gguf"):
            model_name = model_name[:-5]
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"
        return pruned_models_dir() / f"{model_name}-pruned"
    parts = [compact_settings_slug(model, config, include_threshold=True)]
    if seed is not None:
        parts.append(f"seed-{seed}")
    return DECISIONS_DIR / "__".join(part for part in parts if part)


def _viewer_relative_path(source_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), source_dir.resolve())).as_posix()


def _wordfreq_html(
    *,
    settings_path: str,
    token_path: str,
    wordfreq_path: str,
) -> str:
    settings_url = json.dumps(settings_path)
    token_url = json.dumps(token_path)
    wordfreq_url = json.dumps(wordfreq_path)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VocabPruner Wordfreq Viewer</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #101418; color: #f4f7fb; }}
    header {{ display: flex; gap: 16px; align-items: center; padding: 14px 18px; border-bottom: 1px solid #2b333d; flex-wrap: wrap; }}
    h1 {{ font-size: 18px; margin: 0; font-weight: 650; }}
    select {{ background: #19212a; color: #f4f7fb; border: 1px solid #536171; border-radius: 6px; padding: 6px 9px; }}
    main {{ padding: 18px; }}
    canvas {{ width: 100%; height: 560px; background: #151b22; border: 1px solid #2b333d; border-radius: 6px; display: block; }}
    #stats {{ display: flex; gap: 18px; flex-wrap: wrap; color: #cbd5e1; font-size: 13px; }}
    #tip {{ min-height: 28px; padding-top: 12px; color: #cbd5e1; font-size: 14px; }}
  </style>
</head>
<body>
  <header>
    <h1>Wordfreq Distribution</h1>
    <select id="language"></select>
    <div id="stats"></div>
  </header>
  <main>
    <canvas id="chart"></canvas>
    <div id="tip"></div>
  </main>
  <script>
    let settings = {{}};
    let tokenCols = {{}};
    let cols = {{}};
    let languages = [];
    let plotted = [];
    let hoverPoint = null;
    const select = document.getElementById('language');
    const metric = document.createElement('select');
    metric.id = 'metric';
    for (const [value, label] of [
      ['zipf', 'Zipf score'],
      ['rank_percent', 'Top percent'],
      ['rank', 'Dictionary rank'],
    ]) {{
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      metric.append(option);
    }}
    select.after(metric);
    const stats = document.getElementById('stats');
    const canvas = document.getElementById('chart');
    const tip = document.getElementById('tip');
    const ctx = canvas.getContext('2d');

    async function loadData() {{
      tip.textContent = 'Loading wordfreq list...';
      const [settingsResponse, tokenResponse, wordfreqResponse] = await Promise.all([
        fetch({settings_url}),
        fetch({token_url}),
        fetch({wordfreq_url}),
      ]);
      if (!settingsResponse.ok || !tokenResponse.ok || !wordfreqResponse.ok) {{
        throw new Error(`settings ${{settingsResponse.status}}, token ${{tokenResponse.status}}, wordfreq ${{wordfreqResponse.status}}`);
      }}
      settings = await settingsResponse.json();
      const tokenFacts = await tokenResponse.json();
      const wordfreq = await wordfreqResponse.json();
      tokenCols = tokenFacts.columns || {{}};
      cols = wordfreq.columns || {{}};
      languages = [...new Set(cols.language || [])].sort();
      select.replaceChildren();
      for (const language of languages) {{
        const option = document.createElement('option');
        option.value = language;
        option.textContent = language;
        select.append(option);
      }}
      if (languages.length) {{
        select.value = languages[0];
      }}
      rebuildPlot();
      updateStats();
      draw();
    }}

    function row(index) {{
      return {{
        tokenId: cols.token_id[index],
        term: cols.term[index],
        mode: cols.mode[index],
        zipf: cols.zipf[index],
        rank: cols.rank[index],
        rankPercent: cols.rank_percent[index],
      }};
    }}

    function tokenTextById() {{
      const lookup = new Map();
      for (let index = 0; index < (tokenCols.token_id || []).length; index++) {{
        const text = tokenCols.decoded_text[index] || tokenCols.raw_token[index] || String(tokenCols.token_id[index]);
        lookup.set(tokenCols.token_id[index], text);
      }}
      return lookup;
    }}

    function metricValue(item) {{
      if (metric.value === 'zipf') return item.zipf;
      if (metric.value === 'rank_percent') return item.rankPercent;
      return item.rank;
    }}

    function metricIsHigherBetter() {{
      return metric.value === 'zipf';
    }}

    function metricLabel() {{
      return metric.options[metric.selectedIndex].textContent;
    }}

    function metricDisplay(value) {{
      if (value == null) return '-';
      if (metric.value === 'zipf') return value.toFixed(3);
      if (metric.value === 'rank_percent') return `${{value.toFixed(4)}}%`;
      return value.toLocaleString();
    }}

    function better(a, b) {{
      if (a == null) return false;
      if (b == null) return true;
      return metricIsHigherBetter() ? a > b : a < b;
    }}

    function rebuildPlot() {{
      const language = select.value;
      const tokens = tokenTextById();
      const byToken = new Map();
      for (let index = 0; index < (cols.language || []).length; index++) {{
        if (cols.language[index] !== language) continue;
        const item = row(index);
        const value = metricValue(item);
        if (value == null) continue;
        const tokenId = item.tokenId;
        if (!byToken.has(tokenId)) {{
          byToken.set(tokenId, {{
            tokenId,
            token: tokens.get(tokenId) || String(tokenId),
            bestValue: null,
            matches: new Map(),
          }});
        }}
        const group = byToken.get(tokenId);
        if (better(value, group.bestValue)) group.bestValue = value;
        const key = `${{item.term}}\\u0000${{item.mode}}`;
        const previous = group.matches.get(key);
        if (!previous || better(value, metricValue(previous))) group.matches.set(key, item);
      }}
      plotted = [...byToken.values()]
        .map(group => ({{
          ...group,
          matches: [...group.matches.values()].sort((a, b) => {{
            const av = metricValue(a);
            const bv = metricValue(b);
            return metricIsHigherBetter() ? bv - av : av - bv;
          }}),
        }}))
        .filter(group => group.bestValue != null)
        .sort((a, b) => metricIsHigherBetter() ? b.bestValue - a.bestValue : a.bestValue - b.bestValue);
    }}

    function dictionarySize(language) {{
      for (let index = 0; index < (cols.language || []).length; index++) {{
        if (cols.language[index] !== language) continue;
        const rank = cols.rank[index];
        const pct = cols.rank_percent[index];
        if (rank != null && pct != null && pct > 0) {{
          return Math.round(rank * 100 / pct);
        }}
      }}
      return null;
    }}

    function updateStats() {{
      const language = select.value;
      const totalRows = (cols.language || []).length;
      const languageRows = (cols.language || []).filter(value => value === language).length;
      const uniqueMatchedWords = new Set(plotted.flatMap(item => item.matches.map(match => match.term))).size;
      const dictionaryRows = dictionarySize(language);
      const settingsText = settings.settings
        ? `scripts ${{settings.settings.keep_scripts.join(', ')}} | syntax ${{settings.settings.syntax_lists.join(', ')}} | symbols ${{settings.settings.symbol_lists.join(', ')}}`
        : '';
      stats.textContent = [
        `${{plotted.length.toLocaleString()}} token words plotted`,
        `${{uniqueMatchedWords.toLocaleString()}} matched words`,
        `${{languageRows.toLocaleString()}} match rows for ${{language || '-'}}`,
        `${{totalRows.toLocaleString()}} total match rows`,
        dictionaryRows == null ? 'dictionary size unknown' : `${{dictionaryRows.toLocaleString()}} wordfreq dictionary entries`,
        settingsText,
      ].filter(Boolean).join(' | ');
    }}

    function draw() {{
      const deviceRatio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * deviceRatio));
      canvas.height = Math.max(1, Math.floor(rect.height * deviceRatio));
      ctx.setTransform(deviceRatio, 0, 0, deviceRatio, 0, 0);
      const w = rect.width, h = rect.height, pad = 42;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = '#334155';
      ctx.strokeRect(pad, pad, w - pad * 1.5, h - pad * 1.7);
      if (!plotted.length) {{
        tip.textContent = 'No matched token words for this language and metric.';
        return;
      }}
      const values = plotted.map(item => item.bestValue);
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);
      const span = Math.max(1e-9, maxValue - minValue);
      const sampleCount = Math.min(600, plotted.length);
      const points = [];
      ctx.beginPath();
      for (let i = 0; i < sampleCount; i++) {{
        const source = Math.floor(i * (plotted.length - 1) / Math.max(1, sampleCount - 1));
        const item = plotted[source];
        const normalized = metricIsHigherBetter()
          ? (item.bestValue - minValue) / span
          : (maxValue - item.bestValue) / span;
        const x = pad + (i / Math.max(1, sampleCount - 1)) * (w - pad * 1.5);
        const y = h - pad * 0.7 - normalized * (h - pad * 1.7);
        points.push({{x, y, item}});
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 2;
      ctx.stroke();
      if (hoverPoint) {{
        const point = points.reduce((best, item) =>
          Math.abs(item.x - hoverPoint.x) < Math.abs(best.x - hoverPoint.x) ? item : best
        , points[0]);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(point.x, pad);
        ctx.lineTo(point.x, h - pad * 0.7);
        ctx.stroke();
        ctx.fillStyle = '#facc15';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }}
      tip.textContent = `${{select.value}} | ${{metricLabel()}} | ${{plotted.length.toLocaleString()}} token words`;
    }}

    canvas.addEventListener('mousemove', event => {{
      if (!plotted.length) return;
      const rect = canvas.getBoundingClientRect();
      const pad = 42;
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left - pad) / (rect.width - pad * 1.5)));
      const group = plotted[Math.min(plotted.length - 1, Math.max(0, Math.round(ratio * (plotted.length - 1))))];
      const matches = group.matches.slice(0, 8).map(match =>
        `${{match.term}} (${{match.mode}}, ${{metricDisplay(metricValue(match))}})`
      ).join('; ');
      hoverPoint = {{x: event.clientX - rect.left}};
      draw();
      tip.textContent = `token "${{group.token}}" | best ${{metricLabel()}} ${{metricDisplay(group.bestValue)}} | matches: ${{matches}}`;
    }});
    canvas.addEventListener('mouseleave', () => {{
      hoverPoint = null;
      draw();
    }});
    select.addEventListener('change', () => {{
      hoverPoint = null;
      rebuildPlot();
      updateStats();
      draw();
    }});
    metric.addEventListener('change', () => {{
      hoverPoint = null;
      rebuildPlot();
      updateStats();
      draw();
    }});
    window.addEventListener('resize', draw);
    loadData().catch(error => {{
      const fileHint = location.protocol === 'file:'
        ? ' Browsers often block imported local JSON from file://; open this directory through a local HTTP server.'
        : '';
      tip.textContent = `Could not load list data: ${{error}}.${{fileHint}}`;
    }});
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
    workers: int = 1,
    derive_decisions: bool = True,
    progress: bool = False,
    recreate_list: bool = False,
) -> PipelineResult:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    cache_dir = token_cache_dir(model, config, cache_root=cache_root)
    settings = list_settings(model, config)
    cached = None if recreate_list else load_cached_facts(cache_dir, settings)
    cache_reused = cached is not None
    if cached is None:
        token_facts, wordfreq_facts = build_token_facts(records, config, workers=workers, progress=progress)
        write_cached_facts(cache_dir, settings, token_facts, wordfreq_facts)
    else:
        token_facts, wordfreq_facts = cached
    arrays = arrays_from_facts(token_facts, wordfreq_facts, config) if derive_decisions else None
    scores = derive_scores(records, token_facts, wordfreq_facts, config, arrays) if arrays is not None else []
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


def cached_create_only_pipeline(model: str, config: ScoreConfig) -> PipelineResult | None:
    cache_dir = token_cache_dir(model, config)
    cached = load_cached_facts(cache_dir, list_settings(model, config))
    if cached is None:
        return None
    token_facts, wordfreq_facts = cached
    return PipelineResult(
        model=model,
        records=[],
        scores=[],
        retained_ids=[],
        pruned_ids=[],
        cache_dir=cache_dir,
        cache_reused=True,
        token_facts=token_facts,
        wordfreq_facts=wordfreq_facts,
        arrays=None,
    )


def token_cache_dir(model: str, config: ScoreConfig, cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else LISTS_DIR
    return root / compact_settings_slug(model, config, include_threshold=False)


def compact_settings_slug(model: str, config: ScoreConfig, include_threshold: bool = True) -> str:
    parts = [
        model_slug(model),
        f"scripts-{value_slug(sorted(config.keep_scripts))}",
        f"syntax-{list_slug(config.syntax_lists)}",
        f"symbols-{list_slug(config.symbol_lists)}",
        f"langs-{value_slug(config.languages)}",
    ]
    if config.custom_dictionaries:
        parts.append(f"dicts-{value_slug(config.custom_dictionaries)}")
    if include_threshold:
        parts.append(threshold_slug(config))
    return "__".join(parts)


def list_settings(model: str, config: ScoreConfig) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "settings": {
            "keep_scripts": compact_script_values(config.keep_scripts),
            "script_lists_fingerprint": script_list_fingerprint(config.keep_scripts),
            "syntax_lists": compact_list_values(config.syntax_lists, syntax_lists()),
            "syntax_lists_fingerprint": syntax_list_fingerprint(config.syntax_lists),
            "symbol_lists": compact_list_values(config.symbol_lists, symbol_lists()),
            "symbol_lists_fingerprint": symbol_list_fingerprint(config.symbol_lists),
            "languages": list(config.languages),
            "no_wordfreq": config.no_wordfreq,
            "custom_dictionaries": list(config.custom_dictionaries),
        },
    }


def load_cached_facts(cache_dir: Path, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    settings_path = cache_dir / "settings.json"
    token_path = cache_dir / "token_facts.json"
    wordfreq_path = cache_dir / "wordfreq_facts.json"
    if not settings_path.exists() or not token_path.exists() or not wordfreq_path.exists():
        return None
    try:
        cached_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached_settings != settings:
        return None
    return (
        json.loads(token_path.read_text(encoding="utf-8")),
        json.loads(wordfreq_path.read_text(encoding="utf-8")),
    )


def write_cached_facts(
    cache_dir: Path,
    settings: dict[str, Any],
    token_facts: dict[str, Any],
    wordfreq_facts: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (cache_dir / "token_facts.json").write_text(json.dumps(token_facts, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (cache_dir / "wordfreq_facts.json").write_text(json.dumps(wordfreq_facts, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def build_token_facts(
    records: list[TokenRecord],
    config: ScoreConfig,
    workers: int = 1,
    progress: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dictionaries = load_dictionaries(list(config.custom_dictionaries))
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
        "syntax_lists": [],
        "symbol_lists": [],
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
    facts = _token_fact_rows(records, config, dictionaries, workers=workers, progress=progress)
    for record, fact, wordfreq_rows in facts:
        for key, value in fact.items():
            token_columns[key].append(value)
        for row in wordfreq_rows:
            for key, value in row.items():
                wordfreq_columns[key].append(value)
    return (
        {"schema_version": CACHE_SCHEMA_VERSION, "columns": token_columns},
        {"schema_version": CACHE_SCHEMA_VERSION, "columns": wordfreq_columns},
    )


def _token_fact_rows(
    records: list[TokenRecord],
    config: ScoreConfig,
    dictionaries: dict[str, set[str]],
    workers: int,
    progress: bool,
) -> list[tuple[TokenRecord, dict[str, Any], list[dict[str, Any]]]]:
    if workers == 1 or len(records) <= 1:
        return [
            _token_fact_row(record, config, dictionaries)
            for record in _progress(records, progress=progress, total=len(records))
        ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = executor.map(
            _token_fact_row_from_args,
            ((record, config, dictionaries) for record in records),
            chunksize=max(1, len(records) // (workers * 8)),
        )
        return list(_progress(rows, progress=progress, total=len(records)))


def _token_fact_row_from_args(args) -> tuple[TokenRecord, dict[str, Any], list[dict[str, Any]]]:
    record, config, dictionaries = args
    return _token_fact_row(record, config, dictionaries)


def _token_fact_row(
    record: TokenRecord,
    config: ScoreConfig,
    dictionaries: dict[str, set[str]],
) -> tuple[TokenRecord, dict[str, Any], list[dict[str, Any]]]:
    fact = deterministic_token_fact(record, config, dictionaries)
    wordfreq_rows: list[dict[str, Any]] = []
    if fact["retain_candidate"] and not fact["protected"] and not config.no_wordfreq:
        for segment_index, segment in enumerate(fact["evidence_segments"]):
            for diagnostic in wordfreq_diagnostics(segment, list(config.languages)):
                wordfreq_rows.append(
                    {
                        "token_id": record.token_id,
                        "segment_index": segment_index,
                        "language": diagnostic.language,
                        "term": diagnostic.term,
                        "mode": diagnostic.mode,
                        "zipf": diagnostic.frequency,
                        "rank": diagnostic.rank,
                        "rank_percent": diagnostic.rank_percent,
                    }
                )
    return record, fact, wordfreq_rows


def deterministic_token_fact(record: TokenRecord, config: ScoreConfig, dictionaries: dict[str, set[str]]) -> dict[str, Any]:
    text = record.decoded_text or record.raw_token
    profile = script_profile(text)
    structural = is_structural_token(text)
    protected = record.is_special or record.is_byte_fallback or structural
    segments = token_segments(text)
    gate = character_gate(text, config, segments)
    syntax_matches = matching_syntax_lists(text, config.syntax_lists)
    symbol_matches = matching_symbol_lists(text, config.symbol_lists)
    word_segments = language_word_segments(segments)
    evidence_segments = language_evidence_segments(word_segments, dictionaries)
    dict_matches_by_segment = [dictionary_matches(segment.text, dictionaries) for segment in evidence_segments]
    dictionary_accepted = [
        bool(matching_syntax_word_lists(segment.text, config.syntax_lists))
        or has_language_evidence(segment.text, segment_matches, [])
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
    if syntax_matches and (protected or not gate.blocked):
        explanations.append(f"syntax lists matched: {', '.join(syntax_matches)}")
    if symbol_matches and (protected or not gate.blocked):
        explanations.append(f"symbol lists matched: {', '.join(symbol_matches)}")
    flat_dict_matches = [match.to_dict() for matches in dict_matches_by_segment for match in matches[:10]]
    if flat_dict_matches:
        top = ", ".join(f"{match['dictionary']}:{match['term']}/{match['mode']}" for match in flat_dict_matches[:3])
        explanations.append(f"dictionary evidence matched: {top}")
    if not protected and not gate.blocked and not word_segments and (syntax_matches or symbol_matches):
        explanations.append("token retained by selected syntax or symbol lists")
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
        "syntax_lists": syntax_matches,
        "symbol_lists": symbol_matches,
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
        elif config.no_wordfreq:
            status = "retain_candidate"
            explanations.append("decision tree: accept because wordfreq filtering is disabled")
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
                syntax_lists=list(token_columns["syntax_lists"][index]),
                symbol_lists=list(token_columns["symbol_lists"][index]),
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


def list_slug(values: set[str]) -> str:
    return value_slug(sorted(values))


def threshold_slug(config: ScoreConfig) -> str:
    if config.no_wordfreq:
        return "no-wordfreq"
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
