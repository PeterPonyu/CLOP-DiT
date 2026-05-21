from pathlib import Path

from setuptools import find_packages, setup

_this_dir = Path(__file__).resolve().parent
_readme_path = _this_dir / "README.md"
_long_description = (
    _readme_path.read_text(encoding="utf-8")
    if _readme_path.exists()
    else "CLOP-DiT: Text-conditioned single-cell latent generation."
)

setup(
    name="clop-dit",
    version="1.0.0",
    author="Zeyu Fu",
    author_email="fuzeyu09@gmail.com",
    maintainer="Zeyu Fu",
    maintainer_email="fuzeyu09@gmail.com",
    description=(
        "CLOP-DiT: Text-conditioned single-cell latent generation via "
        "contrastive language-omics pretraining and diffusion transformers."
    ),
    long_description=_long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PeterPonyu/CLOP-DiT",
    project_urls={
        "Source": "https://github.com/PeterPonyu/CLOP-DiT",
        "Issue Tracker": "https://github.com/PeterPonyu/CLOP-DiT/issues",
    },
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "scanpy>=1.9.0",
        "anndata>=0.9.0",
        "scib>=1.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "umap-learn>=0.5.0",
        "Pillow>=9.0.0",
        "torchdiffeq>=0.2.3",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "GEOparse>=2.0.3",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "clopdit-generate=src.cli:generate_main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
