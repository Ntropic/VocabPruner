from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path
import json
import os
import re
import socket
import subprocess
import sys

from vocabpruner.analysis.cache import compact_settings_slug
from vocabpruner.analysis.scoring import ScoreConfig
from vocabpruner.presentation.reports import render_jsonl
from vocabpruner.storage.paths import DECISIONS_DIR, pruned_models_dir


def write_wordfreq_html(
    cache_dir: Path,
    output_dir: str | Path | None = None,
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
    scores=None,
    cache_dir: Path | None = None,
    html_dir: str | Path | None = None,
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
            output_dir=html_output_dir,
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
    suffix: str | None = None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    if command == "prune":
        model_name = Path(model).name
        if model_name.lower().endswith(".gguf"):
            model_name = model_name[:-5]
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-") or "model"
        suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix or "").strip("-")
        return pruned_models_dir() / f"{model_name}-{suffix or 'pruned'}"
    parts = [compact_settings_slug(model, config, include_threshold=True)]
    if seed is not None:
        parts.append(f"seed-{seed}")
    return DECISIONS_DIR / "__".join(part for part in parts if part)


def _viewer_relative_path(source_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), source_dir.resolve())).as_posix()


def _wordfreq_html(*, settings_path: str, token_path: str, wordfreq_path: str) -> str:
    template = resources.files("vocabpruner.presentation").joinpath("viewer.html").read_text(encoding="utf-8")
    urls = {"SETTINGS": settings_path, "TOKEN": token_path, "WORDFREQ": wordfreq_path}
    return re.sub(
        r"__VOCABPRUNER_(SETTINGS|TOKEN|WORDFREQ)_URL__",
        lambda match: json.dumps(urls[match.group(1)]).replace("<", "\\u003c"),
        template,
    )


def start_html_server(directory: Path) -> tuple[str, int]:
    port = free_local_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
        ],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return f"http://127.0.0.1:{port}/wordfreq_viewer.html", int(process.pid)


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])
