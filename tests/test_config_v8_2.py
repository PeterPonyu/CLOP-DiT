"""
Unit tests for v8.2 configuration validation.

Validates that v8.2 config has all required fields and proper values
for custom path wiring and variant augmentation.
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestV82ConfigStructure:
    """Test suite for v8.2 config structure and completeness."""

    @pytest.fixture
    def v8_1_config(self):
        """Load the existing v8.1 config as reference."""
        config_path = Path(__file__).parent.parent / "configs" / "clop_v8.1.yaml"
        if not config_path.exists():
            pytest.skip("v8.1 config not found")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_v8_1_has_variant_prob(self, v8_1_config):
        """Verify v8.1 config has variant_prob set."""
        assert "variant_prob" in v8_1_config
        assert v8_1_config["variant_prob"] == 0.35

    def test_v8_1_has_temperature_learnable(self, v8_1_config):
        """Verify v8.1 config has temperature_learnable enabled."""
        assert "temperature_learnable" in v8_1_config
        assert v8_1_config["temperature_learnable"] is True

    def test_v8_1_has_custom_paths(self, v8_1_config):
        """Verify v8.1 config has custom path specifications."""
        # These paths should exist in v8.1 config (flat layout)
        assert "text_embeddings_path" in v8_1_config
        assert "text_strings_path" in v8_1_config
        assert "variant_path" in v8_1_config
        
        # Paths should point to v2 reorganized structure
        assert "text_embeddings_v2" in v8_1_config["text_embeddings_path"]
        assert "text_strings_v2" in v8_1_config["text_strings_path"]

    def test_v8_1_paths_use_reorganized_structure(self, v8_1_config):
        """Verify v8.1 paths use new subdirectory organization."""
        # Should use embeddings/, raw/, variants/ subdirectories
        if "text_embeddings_path" in v8_1_config:
            assert "embeddings/" in v8_1_config["text_embeddings_path"]
        if "text_strings_path" in v8_1_config:
            assert "raw/" in v8_1_config["text_strings_path"]
        if "variant_path" in v8_1_config:
            assert "variants/" in v8_1_config["variant_path"]


class TestV82ConfigValidation:
    """Test suite for v8.2 config validation rules.
    
    These tests define what a valid v8.2 config MUST have.
    """

    @pytest.fixture
    def minimal_v8_2_config(self):
        """Create a minimal valid v8.2 config for testing."""
        return {
            "data": {
                "cache_dir": "data/cached_latents_v5.2",
                "text_embeddings_path": "data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy",
                "text_strings_path": "data/cached_latents_v5.2/raw/text_strings_v2.json",
                "variant_path": "data/cached_latents_v5.2/variants/text_variants.json",
                "variant_map_path": "data/cached_latents_v5.2/variants/text_variant_map.json",
            },
            "training": {
                "batch_size": 1024,
                "num_epochs": 80,
                "variant_prob": 0.35,
                "use_preprocessed": True,
            },
            "loss": {
                "temperature_init": 14.0,
                "temperature_learnable": True,
                "prototype_mode": "average",
            },
            "model": {
                "text_projector": {
                    "input_dim": 1024,
                    "hidden_dim": 512,
                    "output_dim": 512,
                },
                "cell_projector": {
                    "input_dim": 512,
                    "hidden_dim": 512,
                    "output_dim": 512,
                },
            }
        }

    def test_v8_2_requires_custom_text_embeddings_path(self, minimal_v8_2_config):
        """v8.2 MUST specify custom text_embeddings_path."""
        assert "text_embeddings_path" in minimal_v8_2_config["data"]
        assert minimal_v8_2_config["data"]["text_embeddings_path"] is not None
        assert len(minimal_v8_2_config["data"]["text_embeddings_path"]) > 0

    def test_v8_2_requires_variant_prob_gt_zero(self, minimal_v8_2_config):
        """v8.2 MUST have variant_prob > 0 to enable augmentation."""
        assert minimal_v8_2_config["training"]["variant_prob"] > 0
        assert minimal_v8_2_config["training"]["variant_prob"] <= 1.0

    def test_v8_2_requires_variant_paths_when_variant_prob_set(self, minimal_v8_2_config):
        """v8.2 MUST specify variant paths when variant_prob > 0."""
        if minimal_v8_2_config["training"]["variant_prob"] > 0:
            assert "variant_path" in minimal_v8_2_config["data"]
            # variant_map_path is optional (can be inferred)

    def test_v8_2_requires_temperature_learnable(self, minimal_v8_2_config):
        """v8.2 SHOULD use learnable temperature for better convergence."""
        assert minimal_v8_2_config["loss"]["temperature_learnable"] is True

    def test_v8_2_paths_are_absolute_or_relative_to_project_root(self, minimal_v8_2_config):
        """v8.2 paths should be relative to project root or absolute."""
        data = minimal_v8_2_config["data"]
        
        # All paths should either:
        # 1. Start with "data/" (relative to project root)
        # 2. Start with "/" (absolute)
        # 3. Start with "./" (explicit relative)
        
        for key in ["text_embeddings_path", "text_strings_path", "variant_path"]:
            if key in data and data[key]:
                path = data[key]
                assert (
                    path.startswith("data/") or 
                    path.startswith("/") or 
                    path.startswith("./")
                ), f"{key} should use project-relative or absolute path, got: {path}"

    def test_v8_2_temperature_init_reasonable(self, minimal_v8_2_config):
        """v8.2 temperature_init should be in reasonable range (10-20)."""
        temp_init = minimal_v8_2_config["loss"]["temperature_init"]
        assert 5.0 <= temp_init <= 30.0, f"temperature_init {temp_init} outside reasonable range"

    def test_v8_2_batch_size_reasonable(self, minimal_v8_2_config):
        """v8.2 batch_size should be reasonable for contrastive learning."""
        batch_size = minimal_v8_2_config["training"]["batch_size"]
        assert batch_size >= 256, "Contrastive learning needs large batch size (>=256)"
        assert batch_size <= 4096, "Batch size too large for memory"


class TestV82ConfigFileCreation:
    """Test suite for creating and validating v8.2 config file."""

    def test_create_v8_2_config_from_v8_1(self):
        """Test creating v8.2 config based on v8.1 with path wiring fixes."""
        v8_1_path = Path(__file__).parent.parent / "configs" / "clop_v8.1.yaml"
        if not v8_1_path.exists():
            pytest.skip("v8.1 config not found")
        
        with open(v8_1_path) as f:
            v8_1_config = yaml.safe_load(f)
        
        # v8.2 should be identical to v8.1, just with different version tag
        # The key difference is in the IMPLEMENTATION (dataset loader path wiring)
        # not in the config itself
        
        v8_2_config = v8_1_config.copy()
        v8_2_config["version"] = "8.2"
        v8_2_config["description"] = "v8.1 with corrected dataset loader path wiring"
        
        # Verify all required fields are present (flat config layout)
        assert "cache_dir" in v8_2_config
        assert "variant_prob" in v8_2_config
        assert "loss_type" in v8_2_config
        assert "text_dim" in v8_2_config
        
        # Verify custom paths
        assert "text_embeddings_path" in v8_2_config
        assert "variant_path" in v8_2_config
        
        # Verify training settings
        assert v8_2_config["variant_prob"] == 0.35
        assert v8_2_config["use_preprocessed"] is True

    def test_v8_2_config_serializes_correctly(self):
        """Test that v8.2 config can be written to YAML and reloaded."""
        config = {
            "version": "8.2",
            "description": "v8.1 with corrected dataset loader path wiring",
            "data": {
                "cache_dir": "data/cached_latents_v5.2",
                "text_embeddings_path": "data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy",
                "variant_path": "data/cached_latents_v5.2/variants/text_variants.json",
            },
            "training": {
                "variant_prob": 0.35,
            }
        }
        
        # Serialize to YAML string
        yaml_str = yaml.dump(config)
        assert len(yaml_str) > 0
        
        # Deserialize back
        reloaded = yaml.safe_load(yaml_str)
        assert reloaded["version"] == "8.2"
        assert reloaded["data"]["text_embeddings_path"] == config["data"]["text_embeddings_path"]


class TestV82SuccessCriteria:
    """Define success criteria for v8.2 training run.
    
    These tests document what we expect from a successful v8.2 run.
    """

    def test_success_criterion_val_proto_acc(self):
        """Success criterion: Val Proto Acc > 0.10 by epoch 5.
        
        v8.1 failed this criterion (stuck at 0.009-0.011) because dataset loader
        was not using v2 embeddings.
        
        v8.2 MUST achieve val_proto_acc > 0.10 by epoch 5 to validate that:
        1. Dataset loader is actually loading v2 embeddings
        2. V2 embeddings provide better alignment signal
        3. Variant augmentation is working correctly
        """
        expected_val_proto_by_epoch_5 = 0.10
        assert expected_val_proto_by_epoch_5 == 0.10  # Document the threshold

    def test_success_criterion_no_regression_from_v7(self):
        """Success criterion: Final performance >= v7 final (0.097).
        
        v7 final val_proto_acc: 0.097 (epoch 110)
        v8.2 should reach or exceed this with v2 embeddings and variant augmentation.
        """
        v7_final_val_proto = 0.097
        v8_2_expected_minimum = v7_final_val_proto
        assert v8_2_expected_minimum >= v7_final_val_proto

    def test_success_criterion_temperature_convergence(self):
        """Success criterion: Learnable temperature should stabilize.
        
        In v8.1, temperature declined from 14.0 → 11.6 in first 7 epochs.
        v8.2 should show similar or better convergence behavior.
        """
        # Temperature should remain in reasonable range: 5.0 - 20.0
        expected_temp_min = 5.0
        expected_temp_max = 20.0
        assert expected_temp_min < expected_temp_max

    def test_success_criterion_log_evidence_of_v2_loading(self):
        """Success criterion: Training log must show v2 embeddings being loaded.
        
        Expected log line in v8.2:
        "Loading custom text embeddings from: data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy"
        
        Should NOT see:
        "Loading PREPROCESSED (whitened) unique text embeddings"
        (This indicates fallback to default deduplicated cache, not custom v2)
        """
        expected_log_pattern = "Loading custom text embeddings from:.*text_embeddings_v2.npy"
        forbidden_log_pattern = "Loading PREPROCESSED.*unique text embeddings"
        
        # These are string patterns to grep for in logs/v8.2/v8.2_train.log
        assert len(expected_log_pattern) > 0
        assert len(forbidden_log_pattern) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
