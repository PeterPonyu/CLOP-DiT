"""baseline_registry.py — Registry and artifact contract for benchmark methods.

This module centralizes the method metadata used by benchmarking, downstream
validation wrappers, and documentation. It keeps the benchmark extensible
without hard-coding path assumptions inside each panel or evaluation script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.paths import RESULTS_DIR


@dataclass(frozen=True)
class MethodCapabilities:
    """Capabilities supported by a benchmarked method."""

    has_embeddings: bool = True
    has_expression: bool = False
    supports_downstream: bool = False
    supports_de: bool = False
    supports_perturbation: bool = False


@dataclass(frozen=True)
class MethodSpec:
    """Static metadata and artifact locations for one method."""

    slug: str
    display_name: str
    family: str
    source: str
    color: str
    capabilities: MethodCapabilities = field(default_factory=MethodCapabilities)
    built_in: bool = False
    root_generated: bool = False

    def method_dir(self, results_dir: Path) -> Path:
        return results_dir / "baselines" / self.slug

    def embeddings_path(self, results_dir: Path) -> Path:
        if self.root_generated:
            return results_dir / "generated_embeddings.npy"
        return self.method_dir(results_dir) / "embeddings.npy"

    def labels_path(self, results_dir: Path) -> Path:
        if self.root_generated:
            return results_dir / "generated_labels.npy"
        return self.method_dir(results_dir) / "labels.npy"

    def expression_path(self, results_dir: Path) -> Path:
        if self.root_generated:
            return results_dir / "generated_expression.npy"
        return self.method_dir(results_dir) / "expression.npy"

    def expression_labels_path(self, results_dir: Path) -> Path:
        if self.root_generated:
            return results_dir / "generated_expression_labels.npy"
        return self.method_dir(results_dir) / "expression_labels.npy"

    def expression_metrics_path(self, results_dir: Path) -> Path:
        if self.root_generated:
            return results_dir / "expression_metrics.json"
        return self.method_dir(results_dir) / "expression_metrics.json"

    def metadata_path(self, results_dir: Path) -> Path:
        return self.method_dir(results_dir) / "metadata.json"

    def artifacts_exist(self, results_dir: Path) -> bool:
        if self.built_in:
            return True
        return self.embeddings_path(results_dir).exists() and self.labels_path(results_dir).exists()


def _cap(
    *,
    has_embeddings: bool = True,
    has_expression: bool = False,
    supports_downstream: bool = False,
    supports_de: bool = False,
    supports_perturbation: bool = False,
) -> MethodCapabilities:
    return MethodCapabilities(
        has_embeddings=has_embeddings,
        has_expression=has_expression,
        supports_downstream=supports_downstream,
        supports_de=supports_de,
        supports_perturbation=supports_perturbation,
    )


DEFAULT_METHOD_SPECS: List[MethodSpec] = [
    MethodSpec(
        slug="clop_dit",
        display_name="CLOP-DiT",
        family="primary",
        source="project",
        color="#1976D2",
        capabilities=_cap(
            has_embeddings=True,
            has_expression=True,
            supports_downstream=True,
            supports_de=True,
            supports_perturbation=True,
        ),
        root_generated=True,
    ),
    MethodSpec(
        slug="embedding_vae",
        display_name="EmbeddingVAE",
        family="learned_baseline",
        source="project",
        color="#00897B",
        capabilities=_cap(has_embeddings=True),
    ),
    MethodSpec(
        slug="scvi_latent",
        display_name="scVI Latent",
        family="external_baseline",
        source="scvi-tools",
        color="#5E35B1",
        capabilities=_cap(has_embeddings=True),
    ),
    MethodSpec(
        slug="gaussian",
        display_name="Gaussian N(μ,σ²I)",
        family="synthetic",
        source="built_in",
        color="#FF7043",
        capabilities=_cap(has_embeddings=True),
        built_in=True,
    ),
    MethodSpec(
        slug="shuffled_labels",
        display_name="Shuffled Labels",
        family="synthetic",
        source="built_in",
        color="#4CAF50",
        capabilities=_cap(has_embeddings=True),
        built_in=True,
    ),
    MethodSpec(
        slug="random_normal",
        display_name="Random N(0,I)",
        family="synthetic",
        source="built_in",
        color="#9C27B0",
        capabilities=_cap(has_embeddings=True),
        built_in=True,
    ),
    MethodSpec(
        slug="mean_only",
        display_name="Mean-only (collapse)",
        family="synthetic",
        source="built_in",
        color="#FFC107",
        capabilities=_cap(has_embeddings=True),
        built_in=True,
    ),
]


def get_method_specs(results_dir: Optional[Path] = None) -> List[MethodSpec]:
    """Return registered methods in plotting / ranking order."""

    _ = results_dir or RESULTS_DIR
    return list(DEFAULT_METHOD_SPECS)


def get_method_map(results_dir: Optional[Path] = None) -> Dict[str, MethodSpec]:
    return {spec.display_name: spec for spec in get_method_specs(results_dir)}


def get_method_by_slug(slug: str, results_dir: Optional[Path] = None) -> Optional[MethodSpec]:
    for spec in get_method_specs(results_dir):
        if spec.slug == slug:
            return spec
    return None


def load_baseline_manifest(results_dir: Optional[Path] = None) -> Dict:
    """Load aggregate baseline manifest if present."""

    rdir = Path(results_dir or RESULTS_DIR)
    manifest_path = rdir / "baselines" / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        return json.load(f)


def load_method_metadata(spec: MethodSpec, results_dir: Optional[Path] = None) -> Dict:
    """Load per-method metadata if present, otherwise return defaults."""

    rdir = Path(results_dir or RESULTS_DIR)
    data = {
        "slug": spec.slug,
        "display_name": spec.display_name,
        "family": spec.family,
        "source": spec.source,
        "color": spec.color,
        "capabilities": {
            "has_embeddings": spec.capabilities.has_embeddings,
            "has_expression": spec.capabilities.has_expression,
            "supports_downstream": spec.capabilities.supports_downstream,
            "supports_de": spec.capabilities.supports_de,
            "supports_perturbation": spec.capabilities.supports_perturbation,
        },
        "artifacts_available": spec.artifacts_exist(rdir),
    }
    meta_path = spec.metadata_path(rdir)
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                on_disk = json.load(f)
            data.update(on_disk)
        except Exception:
            pass
    return data


def expected_artifact_contract() -> Dict[str, str]:
    """Return the standard artifact contract for baseline methods."""

    return {
        "embeddings": "results/baselines/{method}/embeddings.npy",
        "labels": "results/baselines/{method}/labels.npy",
        "expression": "results/baselines/{method}/expression.npy",
        "expression_labels": "results/baselines/{method}/expression_labels.npy",
        "metadata": "results/baselines/{method}/metadata.json",
        "expression_metrics": "results/baselines/{method}/expression_metrics.json",
        "downstream": "results/downstream/{method}_*.json",
    }
