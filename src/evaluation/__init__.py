# evaluation/__init__.py
from .metrics import GenerationMetrics
from .visualizer import EmbeddingVisualizer
from .baseline_registry import get_method_specs
from .downstream_biology import run_all_downstream, build_matched_adata
from .model_benchmarking import run_benchmark
