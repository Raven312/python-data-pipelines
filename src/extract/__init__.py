"""Extract module - Data ingestion from various sources."""

from .extractor import DataExtractor, extract_from_dict, extract_from_csv, extract_from_api

__all__ = ["DataExtractor", "extract_from_dict", "extract_from_csv", "extract_from_api"]
