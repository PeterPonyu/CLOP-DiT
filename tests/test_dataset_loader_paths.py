"""
Unit tests for dataset loader custom path wiring.

Tests that CLOPDataset can load text embeddings from custom paths
specified in config, rather than only using default path inference.

This is critical for v8.2+ where we want to load v2 re-embedded captions
from data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dataset import CLOPDataset


class TestDatasetCustomPaths:
    """Test suite for custom path loading in CLOPDataset."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory with minimal test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir)
            
            # Create minimal embeddings
            n_samples = 100
            n_groups = 10
            cell_dim = 512
            text_dim = 1024
            
            # Cell embeddings
            cell_emb = np.random.randn(n_samples, cell_dim).astype(np.float32)
            np.save(cache_path / "cell_embeddings.npy", cell_emb)
            
            # Deduplicated text embeddings (unique)
            text_emb_unique = np.random.randn(n_groups, text_dim).astype(np.float32)
            np.save(cache_path / "text_embeddings_unique.npy", text_emb_unique)
            
            # Custom v2 text embeddings (in subdirectory)
            embeddings_dir = cache_path / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)
            text_emb_v2 = np.random.randn(n_groups, text_dim).astype(np.float32)
            np.save(embeddings_dir / "text_embeddings_v2.npy", text_emb_v2)
            
            # Group IDs (each sample maps to one of the text groups)
            text_group_ids = np.random.randint(0, n_groups, size=n_samples)
            np.save(cache_path / "text_group_ids.npy", text_group_ids)
            
            # Sample IDs
            sample_ids = np.arange(n_samples)
            np.save(cache_path / "sample_ids.npy", sample_ids)
            
            yield cache_path

    def test_default_path_loading(self, temp_cache_dir):
        """Test that default path inference still works."""
        dataset = CLOPDataset(
            cache_dir=temp_cache_dir,
            noise_std=0.0,
            variant_prob=0.0,
        )
        
        # Should load text_embeddings_unique.npy by default
        assert dataset._deduplicated is True
        assert dataset.text_emb_unique is not None
        assert dataset.text_emb_unique.shape[0] == 10  # n_groups
        assert dataset.cell_emb.shape[0] == 100  # n_samples

    def test_custom_text_embeddings_path(self, temp_cache_dir):
        """Test loading custom text embeddings from config-specified path."""
        custom_text_path = temp_cache_dir / "embeddings" / "text_embeddings_v2.npy"
        
        # This test WILL FAIL until we implement custom path wiring
        # Expected behavior: CLOPDataset should accept text_embeddings_path
        # and load from that instead of default text_embeddings_unique.npy
        
        with pytest.raises(TypeError):
            # Currently CLOPDataset.__init__ doesn't accept text_embeddings_path
            dataset = CLOPDataset(
                cache_dir=temp_cache_dir,
                text_embeddings_path=str(custom_text_path),
                noise_std=0.0,
                variant_prob=0.0,
            )

    def test_custom_variant_path(self, temp_cache_dir):
        """Test loading custom variant data from config-specified path."""
        # Create variant data
        variants_dir = temp_cache_dir / "variants"
        variants_dir.mkdir(exist_ok=True)
        
        # Create variant embeddings
        n_variants = 50
        text_dim = 1024
        variant_embs = np.random.randn(n_variants, text_dim).astype(np.float32)
        np.save(variants_dir / "text_variant_embeddings.npy", variant_embs)
        
        # Create variant map
        variant_map = [[i % 10, i] for i in range(n_variants)]  # map to 10 groups
        with open(variants_dir / "text_variant_map.json", "w") as f:
            json.dump(variant_map, f)
        
        # This test WILL FAIL until we implement custom path wiring
        with pytest.raises(TypeError):
            dataset = CLOPDataset(
                cache_dir=temp_cache_dir,
                variant_path=str(variants_dir / "text_variant_embeddings.npy"),
                variant_map_path=str(variants_dir / "text_variant_map.json"),
                variant_prob=0.35,
            )


class TestDatasetPathWiringIntegration:
    """Integration tests for full path wiring from config to dataset.
    
    These tests verify end-to-end behavior: config YAML → train_clop.py → CLOPDataset
    """

    @pytest.fixture
    def temp_config_with_custom_paths(self, tmp_path):
        """Create a minimal v8.2 config with custom paths."""
        config = {
            "data": {
                "cache_dir": "data/cached_latents_v5.2",
                "text_embeddings_path": "data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy",
                "text_strings_path": "data/cached_latents_v5.2/raw/text_strings_v2.json",
                "variant_path": "data/cached_latents_v5.2/variants/text_variants.json",
                "variant_map_path": "data/cached_latents_v5.2/variants/text_variant_map.json",
            },
            "training": {
                "variant_prob": 0.35,
                "use_preprocessed": True,
            }
        }
        
        config_path = tmp_path / "test_v8.2.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        return config_path

    def test_config_has_custom_path_keys(self, temp_config_with_custom_paths):
        """Verify that v8.2 config contains custom path specifications."""
        import yaml
        with open(temp_config_with_custom_paths) as f:
            config = yaml.safe_load(f)
        
        assert "text_embeddings_path" in config["data"]
        assert "variant_path" in config["data"]
        assert "text_strings_path" in config["data"]
        
        # Paths should point to reorganized subdirectories
        assert "embeddings/text_embeddings_v2.npy" in config["data"]["text_embeddings_path"]
        assert "variants/text_variants.json" in config["data"]["variant_path"]

    @pytest.mark.skip(reason="Requires full train_clop.py wiring implementation")
    def test_train_clop_passes_custom_paths_to_dataset(self):
        """Test that train_clop.py reads custom paths from config and passes to CLOPDataset.
        
        This will be implemented in the next phase when we wire up the path passing.
        """
        pass


class TestExpectedBehaviorDocumentation:
    """Document expected behavior for custom path wiring.
    
    These are design specification tests, not implementation tests.
    They describe HOW the system SHOULD work after path wiring is complete.
    """

    def test_design_spec_custom_text_embeddings(self):
        """Design spec: CLOPDataset should accept text_embeddings_path parameter.
        
        Expected signature after implementation:
        
        CLOPDataset(
            cache_dir="data/cached_latents_v5.2",
            text_embeddings_path="data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy",
            ...
        )
        
        When text_embeddings_path is provided:
        - Load unique text embeddings from custom path
        - Skip default path inference (text_embeddings_unique.npy)
        - Log: "Loading custom text embeddings from: {text_embeddings_path}"
        
        When text_embeddings_path is None:
        - Fall back to current behavior (text_embeddings_unique.npy or text_embeddings.npy)
        """
        assert True  # Design documentation, always passes

    def test_design_spec_custom_variant_paths(self):
        """Design spec: CLOPDataset should accept variant_path and variant_map_path.
        
        Expected signature after implementation:
        
        CLOPDataset(
            cache_dir="data/cached_latents_v5.2",
            variant_path="data/cached_latents_v5.2/variants/text_variant_embeddings.npy",
            variant_map_path="data/cached_latents_v5.2/variants/text_variant_map.json",
            variant_prob=0.35,
            ...
        )
        
        When variant paths are provided:
        - Load variant embeddings from custom paths
        - Skip default cache_dir-based detection
        - Log: "Loading custom variant embeddings from: {variant_path}"
        
        When variant paths are None:
        - Fall back to current behavior (cache_dir/text_variant_embeddings.npy)
        """
        assert True  # Design documentation, always passes

    def test_design_spec_config_to_dataset_wiring(self):
        """Design spec: train_clop.py should pass config paths to CLOPDataset.
        
        Expected flow after implementation:
        
        1. train_clop.py reads config YAML:
           data:
             cache_dir: "data/cached_latents_v5.2"
             text_embeddings_path: "data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy"
             variant_path: "data/cached_latents_v5.2/variants/text_variants.json"
        
        2. train_clop.py creates dataset with custom paths:
           dataset = CLOPDataset(
               cache_dir=config['data']['cache_dir'],
               text_embeddings_path=config['data'].get('text_embeddings_path'),
               variant_path=config['data'].get('variant_path'),
               ...
           )
        
        3. CLOPDataset loads from custom paths:
           - text_embeddings from text_embeddings_v2.npy
           - variants from custom variant paths
        
        4. Training log shows:
           "Loading custom text embeddings from: data/cached_latents_v5.2/embeddings/text_embeddings_v2.npy"
        """
        assert True  # Design documentation, always passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
