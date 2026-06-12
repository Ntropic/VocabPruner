from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenpruner.hf_cache import mark_analysis_download
from tokenpruner.paths import HF_MODELS_DIR


@dataclass(frozen=True)
class ModelArchitectureReport:
    model_type: str
    architectures: tuple[str, ...]
    vocabulary_surfaces: tuple[tuple[str, int], ...]
    encoders: tuple[str, ...]
    mtp_settings: tuple[tuple[str, Any], ...]
    mtp_head_status: str
    embedding_ties: tuple[tuple[str, bool], ...]
    mtp_dedicated_embeddings: bool | None
    quantization_method: str | None

    @property
    def has_mtp(self) -> bool:
        return any(_enabled(value) for _, value in self.mtp_settings)


def load_model_config(model: str, trust_remote_code: bool = False, **kwargs: Any) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to inspect model architecture. Install project dependencies first."
        ) from exc

    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        cache_dir=HF_MODELS_DIR,
        **kwargs,
    )
    mark_analysis_download(model, trust_remote_code)
    return config


def inspect_model_architecture(config: Any) -> ModelArchitectureReport:
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    flattened = tuple(_flatten(data))
    model_type = str(data.get("model_type") or getattr(config, "model_type", "unknown"))
    architectures = tuple(str(value) for value in data.get("architectures") or ())

    vocabulary_surfaces = tuple(
        (path, int(value))
        for path, value in flattened
        if _is_vocabulary_size(path, value)
    )
    encoders = tuple(sorted(_encoder_paths(data)))
    mtp_settings = tuple(
        (path, value)
        for path, value in flattened
        if _is_mtp_setting(path, value)
    )
    embedding_ties = tuple(
        (path.rsplit(".", 1)[0] if "." in path else "root", bool(value))
        for path, value in flattened
        if path.split(".")[-1] == "tie_word_embeddings" and isinstance(value, bool)
    )
    mtp_dedicated_embeddings = next(
        (
            bool(value)
            for path, value in flattened
            if path.split(".")[-1] == "mtp_use_dedicated_embeddings" and isinstance(value, bool)
        ),
        None,
    )
    quantization_method = next(
        (
            str(value)
            for path, value in flattened
            if path == "quantization_config.quant_method"
        ),
        None,
    )

    mtp_head_declared = any(
        _is_mtp_path(path)
        and path.split(".")[-1] in {"lm_head", "output_head"}
        and _enabled(value)
        for path, value in flattened
    )
    mtp_vocabulary_declared = any(
        _is_mtp_path(path)
        and path.split(".")[-1] in {"vocab_size", "codebook_size"}
        for path, _ in flattened
    )
    if not any(_enabled(value) for _, value in mtp_settings):
        mtp_head_status = "not applicable"
    elif mtp_head_declared:
        mtp_head_status = "declared in config"
    elif mtp_vocabulary_declared:
        mtp_head_status = "separate vocabulary declared; head sharing unknown"
    else:
        mtp_head_status = "not declared; inspect model modules or checkpoint tensors"

    return ModelArchitectureReport(
        model_type=model_type,
        architectures=architectures,
        vocabulary_surfaces=vocabulary_surfaces,
        encoders=encoders,
        mtp_settings=mtp_settings,
        mtp_head_status=mtp_head_status,
        embedding_ties=embedding_ties,
        mtp_dedicated_embeddings=mtp_dedicated_embeddings,
        quantization_method=quantization_method,
    )


def render_model_architecture(report: ModelArchitectureReport) -> str:
    lines = [
        "Model Architecture",
        f"Model Type: {report.model_type}",
        f"Architectures: {', '.join(report.architectures) if report.architectures else 'not declared'}",
        f"MTP: {'yes' if report.has_mtp else 'no'}",
        f"Separate MTP LM Head: {report.mtp_head_status}",
        "MTP Embeddings: "
        + (
            "dedicated"
            if report.mtp_dedicated_embeddings is True
            else "shared"
            if report.mtp_dedicated_embeddings is False
            else "not declared"
        ),
        f"Quantization: {report.quantization_method or 'none declared'}",
    ]
    if report.mtp_settings:
        lines.append("MTP Settings:")
        lines.extend(f"- {path}: {value}" for path, value in report.mtp_settings)
    lines.append("Vocabulary-Bearing Surfaces:")
    lines.extend(
        f"- {path}: {size}"
        for path, size in report.vocabulary_surfaces
    )
    if not report.vocabulary_surfaces:
        lines.append("- none declared")
    lines.append("Declared Encoders:")
    lines.extend(f"- {path}" for path in report.encoders)
    if not report.encoders:
        lines.append("- none declared")
    lines.append("Embedding / LM-Head Tying:")
    lines.extend(
        f"- {path}: {'tied' if tied else 'untied'}"
        for path, tied in report.embedding_ties
    )
    if not report.embedding_ties:
        lines.append("- not declared")
    if len(report.vocabulary_surfaces) > 1 or report.has_mtp:
        lines.append("Pruning Note: resize every affected embedding and output head; preserve modality/control IDs.")
    return "\n".join(lines)


def _flatten(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child, path)
        return
    yield prefix, value


def _encoder_paths(data: dict[str, Any], prefix: str = "") -> set[str]:
    encoders: set[str] = set()
    for key, value in data.items():
        key_text = str(key)
        path = f"{prefix}.{key_text}" if prefix else key_text
        lowered = key_text.lower()
        if isinstance(value, dict):
            if any(marker in lowered for marker in ("encoder", "vision", "visual", "audio", "speech", "image", "video")):
                encoders.add(path)
            encoders.update(_encoder_paths(value, path))
    return encoders


def _is_vocabulary_size(path: str, value: Any) -> bool:
    key = path.split(".")[-1]
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and (key == "vocab_size" or key.endswith("_vocab_size") or key == "codebook_size")
    )


def _is_mtp_setting(path: str, value: Any) -> bool:
    lowered = path.lower()
    key = lowered.split(".")[-1]
    return (
        _is_mtp_path(path)
        or key in {"num_next_tokens", "num_prediction_heads"}
    ) and isinstance(value, (bool, int, str))


def _is_mtp_path(path: str) -> bool:
    lowered = path.lower()
    return "mtp" in lowered or "multi_token" in lowered or "nextn" in lowered


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        return value.lower() not in {"", "0", "false", "none", "disabled"}
    return False
