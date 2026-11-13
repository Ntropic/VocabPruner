from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PruneResult:
    output_dir: Path
    retained_count: int
    pruned_count: int
    old_to_new: dict[int, int]
    new_to_old: dict[int, int]
    transformed_surfaces: tuple["TransformedSurface", ...] = ()
    original_checkpoint_bytes: int | None = None
    pruned_checkpoint_bytes: int | None = None


@dataclass(frozen=True)
class TransformedSurface:
    name: str
    kind: str
    shared_with: str | None
    tensors: tuple[str, ...]
    quantized: bool


@dataclass(frozen=True)
class TokenRecord:
    token_id: int
    raw_token: str
    decoded_text: str
    is_special: bool = False
    is_byte_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptProfile:
    counts: dict[str, int]
    total_meaningful: int
    low_confidence: bool

    @property
    def scripts(self) -> list[str]:
        return sorted(self.counts)

    def ratio(self, script: str) -> float:
        if self.total_meaningful == 0:
            return 0.0
        return self.counts.get(script, 0) / self.total_meaningful

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "scripts": self.scripts,
            "total_meaningful": self.total_meaningful,
            "low_confidence": self.low_confidence,
        }


@dataclass(frozen=True)
class DictionaryMatch:
    dictionary: str
    term: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WordfreqMatch:
    language: str
    term: str
    mode: str
    frequency: float
    rank: int | None = None
    rank_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenScore:
    token: TokenRecord
    script_profile: ScriptProfile
    protected: bool
    status: str
    syntax_lists: list[str] = field(default_factory=list)
    symbol_lists: list[str] = field(default_factory=list)
    dictionary_matches: list[DictionaryMatch] = field(default_factory=list)
    wordfreq_matches: list[WordfreqMatch] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token.to_dict(),
            "script_profile": self.script_profile.to_dict(),
            "protected": self.protected,
            "status": self.status,
            "syntax_lists": list(self.syntax_lists),
            "symbol_lists": list(self.symbol_lists),
            "dictionary_matches": [match.to_dict() for match in self.dictionary_matches],
            "wordfreq_matches": [match.to_dict() for match in self.wordfreq_matches],
            "explanations": list(self.explanations),
        }
