"""Tests for current production configs (clop, dit)."""

import pytest
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestClopConfig:
    """Validate the production CLOP config."""

    @pytest.fixture
    def config(self):
        path = CONFIGS_DIR / "clop.yaml"
        if not path.exists():
            pytest.skip("clop.yaml not found")
        with open(path) as f:
            return yaml.safe_load(f)

    REQUIRED_KEYS = [
        "cache_dir", "val_split", "split_strategy",
        "text_dim", "cell_dim", "proj_dim",
        "dropout", "temperature",
    ]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, config, key):
        assert key in config, f"Missing required key: {key}"

    def test_stratified_split(self, config):
        assert config["split_strategy"] == "stratified"

    def test_proj_dim_matches_dit(self, config):
        dit_path = CONFIGS_DIR / "dit.yaml"
        if not dit_path.exists():
            pytest.skip("dit.yaml not found")
        with open(dit_path) as f:
            dit = yaml.safe_load(f)
        assert config["proj_dim"] == dit["cond_dim"], (
            f"CLOP proj_dim ({config['proj_dim']}) != DiT cond_dim ({dit['cond_dim']})"
        )


class TestDitConfig:
    """Validate the production DiT config."""

    @pytest.fixture
    def config(self):
        path = CONFIGS_DIR / "dit.yaml"
        if not path.exists():
            pytest.skip("dit.yaml not found")
        with open(path) as f:
            return yaml.safe_load(f)

    REQUIRED_KEYS = [
        "cache_dir", "latent_dim", "hidden_dim", "cond_dim",
        "num_blocks", "num_heads", "num_tokens",
        "cond_drop_prob", "split_strategy",
    ]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, config, key):
        assert key in config, f"Missing required key: {key}"

    def test_num_tokens_divides_latent_dim(self, config):
        assert config["latent_dim"] % config["num_tokens"] == 0, (
            "latent_dim must be divisible by num_tokens"
        )

    def test_cond_drop_positive(self, config):
        assert 0 < config["cond_drop_prob"] < 1
