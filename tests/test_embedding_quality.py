"""Tests for embedding quality metrics used in CLOP training monitoring."""

import torch
import pytest
from src.evaluation.embedding_quality import (
    compute_alignment,
    compute_uniformity,
    compute_mean_cosine_sim,
    compute_group_quality,
    compute_all_quality_metrics,
)


class TestAlignment:
    """Wang & Isola alignment metric."""

    def test_perfect_alignment_is_zero(self):
        """Identical text and cell projections → alignment = 0."""
        proj = torch.randn(32, 128)
        proj = torch.nn.functional.normalize(proj, dim=-1)
        assert compute_alignment(proj, proj) == pytest.approx(0.0, abs=1e-5)

    def test_random_alignment_is_positive(self):
        """Unrelated projections → alignment > 0."""
        text = torch.nn.functional.normalize(torch.randn(64, 128), dim=-1)
        cell = torch.nn.functional.normalize(torch.randn(64, 128), dim=-1)
        assert compute_alignment(text, cell) > 0.5

    def test_orthogonal_embeddings(self):
        """Orthogonal pairs have alignment ≈ 2.0 (max L2² for unit vectors)."""
        text = torch.zeros(2, 4)
        text[0, 0] = 1.0
        text[1, 1] = 1.0
        cell = torch.zeros(2, 4)
        cell[0, 2] = 1.0
        cell[1, 3] = 1.0
        assert compute_alignment(text, cell) == pytest.approx(2.0, abs=1e-5)


class TestUniformity:
    """Wang & Isola uniformity metric."""

    def test_clustered_embeddings_high_uniformity(self):
        """Clustered embeddings → uniformity close to 0 (bad)."""
        # All embeddings nearly identical → exp(-t*0) = 1 → log(1) = 0
        base = torch.nn.functional.normalize(torch.randn(1, 64), dim=-1)
        emb = base.expand(50, -1) + torch.randn(50, 64) * 0.01
        emb = torch.nn.functional.normalize(emb, dim=-1)
        u = compute_uniformity(emb)
        assert u > -0.5  # close to 0 (bad uniformity)

    def test_spread_embeddings_low_uniformity(self):
        """Well-spread random embeddings → uniformity << 0 (good)."""
        emb = torch.nn.functional.normalize(torch.randn(200, 128), dim=-1)
        u = compute_uniformity(emb)
        assert u < -1.0  # well below 0 (good uniformity)


class TestMeanCosineSim:
    """CLOP score (mean cosine similarity)."""

    def test_identical_gives_one(self):
        proj = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
        assert compute_mean_cosine_sim(proj, proj) == pytest.approx(1.0, abs=1e-5)

    def test_random_is_near_zero(self):
        text = torch.nn.functional.normalize(torch.randn(500, 128), dim=-1)
        cell = torch.nn.functional.normalize(torch.randn(500, 128), dim=-1)
        sim = compute_mean_cosine_sim(text, cell)
        assert abs(sim) < 0.2  # random → near zero


class TestGroupQuality:
    """Inter-group separation and intra-group cohesion."""

    def test_well_separated_groups(self):
        """Distinct groups should have high separation."""
        D = 64
        # Two groups: orthogonal centroids
        text = torch.zeros(20, D)
        cell = torch.zeros(20, D)
        group_ids = torch.zeros(20, dtype=torch.long)

        # Group 0: along dim 0
        text[:10, 0] = 1.0
        cell[:10, 0] = 1.0
        cell[:10] += torch.randn(10, D) * 0.05
        group_ids[:10] = 0

        # Group 1: along dim 1
        text[10:, 1] = 1.0
        cell[10:, 1] = 1.0
        cell[10:] += torch.randn(10, D) * 0.05
        group_ids[10:] = 1

        text = torch.nn.functional.normalize(text, dim=-1)
        cell = torch.nn.functional.normalize(cell, dim=-1)

        metrics = compute_group_quality(text, cell, group_ids)
        assert metrics["inter_sep"] > 0.8  # nearly orthogonal
        assert metrics["intra_cohesion"] > 0.9  # tight clusters
        assert metrics["text_cell_align"] > 0.8  # text↔cell aligned

    def test_single_group(self):
        """Single group returns defaults."""
        text = torch.nn.functional.normalize(torch.randn(10, 32), dim=-1)
        cell = torch.nn.functional.normalize(torch.randn(10, 32), dim=-1)
        group_ids = torch.zeros(10, dtype=torch.long)
        metrics = compute_group_quality(text, cell, group_ids)
        assert metrics["inter_sep"] == 0.0  # can't compute separation


class TestComputeAll:
    """Integration: compute_all_quality_metrics returns all keys."""

    def test_returns_all_keys_with_groups(self):
        text = torch.nn.functional.normalize(torch.randn(50, 64), dim=-1)
        cell = torch.nn.functional.normalize(torch.randn(50, 64), dim=-1)
        group_ids = torch.randint(0, 5, (50,))
        m = compute_all_quality_metrics(text, cell, group_ids)
        expected = {
            "alignment", "uniformity_text", "uniformity_cell",
            "mean_cosine_sim", "inter_sep", "intra_cohesion", "text_cell_align",
        }
        assert expected.issubset(m.keys())

    def test_returns_core_keys_without_groups(self):
        text = torch.nn.functional.normalize(torch.randn(50, 64), dim=-1)
        cell = torch.nn.functional.normalize(torch.randn(50, 64), dim=-1)
        m = compute_all_quality_metrics(text, cell, group_ids=None)
        assert "alignment" in m
        assert "uniformity_text" in m
        assert "mean_cosine_sim" in m
        assert "inter_sep" not in m  # no group_ids → no group metrics
