from __future__ import annotations

import importlib.util
import subprocess
import sys

from vocabpruner.checkpoints.architecture import ModelArchitectureReport
from vocabpruner.errors import PruneRefusal


def validate_pruning_runtime(report: ModelArchitectureReport) -> None:
    missing = missing_pruning_packages(report)
    if missing:
        raise PruneRefusal("Missing pruning runtime package(s): " + ", ".join(missing))


def missing_pruning_packages(report: ModelArchitectureReport) -> tuple[str, ...]:
    missing: list[str] = []
    if not _module_available("torch"):
        missing.append("torch")
    method = (report.quantization_method or "").lower()
    if method == "modelopt" and not _module_available("modelopt"):
        missing.append("nvidia-modelopt")
    if method in {"compressed-tensors", "compressed_tensors"} and not _module_available("compressed_tensors"):
        missing.append("compressed-tensors")
    return tuple(missing)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_pruning_runtime(report, assume_yes: bool = False) -> None:
    missing = missing_pruning_packages(report)
    if not missing:
        return

    command = [sys.executable, "-m", "pip", "install", *missing]
    print("Missing packages required to prune this checkpoint:")
    for package in missing:
        print(f"- {package}")
    print("Suggested command:")
    print(" ".join(command))

    if not assume_yes:
        try:
            answer = input("Install these packages now? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            raise SystemExit("Required pruning packages were not installed.")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Package installation failed with exit code {exc.returncode}.") from exc

    try:
        validate_pruning_runtime(report)
    except PruneRefusal as exc:
        raise SystemExit(f"Packages were installed, but runtime validation still failed: {exc}") from exc
