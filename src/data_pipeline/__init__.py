# data_pipeline/__init__.py
# Lazy imports to avoid heavy dependencies (scanpy) when not needed
from .dataset import CLOPDataset, DiTDataset, InferenceDataset

def __getattr__(name):
    if name == "GEOFetcher":
        from .geo_fetcher import GEOFetcher
        return GEOFetcher
    if name == "TextCleaner":
        from .text_cleaner import TextCleaner
        return TextCleaner
    if name == "LatentCacheBuilder":
        from .cache_builder import LatentCacheBuilder
        return LatentCacheBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
